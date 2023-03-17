from __future__ import annotations

import os
from logging import getLogger
from typing import Any

import librosa
import numpy as np
import torch
from cm_time import timer

from so_vits_svc_fork import cluster, utils
from so_vits_svc_fork.inference import slicer
from so_vits_svc_fork.models import SynthesizerTrn

from ..utils import HUBERT_SAMPLING_RATE

LOG = getLogger(__name__)


def pad_array(arr, target_length):
    current_length = arr.shape[0]
    if current_length >= target_length:
        return arr[
            (current_length - target_length)
            // 2 : (current_length - target_length)
            // 2
            + target_length,
            ...,
        ]
    else:
        pad_width = target_length - current_length
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        padded_arr = np.pad(
            arr, (pad_left, pad_right), "constant", constant_values=(0, 0)
        )
        return padded_arr


class Svc:
    def __init__(
        self,
        *,
        net_g_path: str,
        config_path: str,
        device: torch.device | str | None = None,
        cluster_model_path: str | None = None,
    ):
        self.net_g_path = net_g_path
        if device is None:
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dev = torch.device(device)
        self.net_g_ms = None
        self.hps_ms = utils.get_hparams_from_file(config_path)
        self.target_sample = self.hps_ms.data.sampling_rate
        self.hop_size = self.hps_ms.data.hop_length
        self.spk2id = self.hps_ms.spk
        self.hubert_model = utils.get_hubert_model().to(self.dev)
        self.load_model()
        if cluster_model_path is not None and os.path.exists(cluster_model_path):
            self.cluster_model = cluster.get_cluster_model(cluster_model_path)

    def load_model(self):
        self.net_g_ms = SynthesizerTrn(
            self.hps_ms.data.filter_length // 2 + 1,
            self.hps_ms.train.segment_size // self.hps_ms.data.hop_length,
            **self.hps_ms.model,
        )
        _ = utils.load_checkpoint(self.net_g_path, self.net_g_ms, None)
        if "half" in self.net_g_path and torch.cuda.is_available():
            _ = self.net_g_ms.half().eval().to(self.dev)
        else:
            _ = self.net_g_ms.eval().to(self.dev)

    def get_unit_f0(
        self,
        audio: np.ndarray[Any, np.dtype[np.float64]],
        tran: int,
        cluster_infer_ratio: float,
        speaker: int | str,
    ):
        f0 = utils.compute_f0_parselmouth(
            audio, sampling_rate=self.target_sample, hop_length=self.hop_size
        )
        f0, uv = utils.interpolate_f0(f0)
        f0 = torch.FloatTensor(f0)
        uv = torch.FloatTensor(uv)
        f0 = f0 * 2 ** (tran / 12)
        f0 = f0.unsqueeze(0).to(self.dev)
        uv = uv.unsqueeze(0).to(self.dev)

        wav16k = librosa.resample(
            audio, orig_sr=self.target_sample, target_sr=HUBERT_SAMPLING_RATE
        )
        wav16k = torch.from_numpy(wav16k).to(self.dev)
        c = utils.get_hubert_content(self.hubert_model, wav_16k_tensor=wav16k)
        c = utils.repeat_expand_2d(c.squeeze(0), f0.shape[1])

        if cluster_infer_ratio != 0:
            cluster_c = cluster.get_cluster_center_result(
                self.cluster_model, c.cpu().numpy().T, speaker
            ).T
            cluster_c = torch.FloatTensor(cluster_c).to(self.dev)
            c = cluster_infer_ratio * cluster_c + (1 - cluster_infer_ratio) * c

        c = c.unsqueeze(0)
        return c, f0, uv

    def infer(
        self,
        speaker: int | str,
        transpose: int,
        audio: np.ndarray[Any, np.dtype[np.float32]],
        cluster_infer_ratio: float = 0,
        auto_predict_f0: bool = False,
        noise_scale: float = 0.4,
    ):
        audio = audio.astype(np.float32)
        # get speaker id
        if isinstance(speaker, int):
            if len(self.spk2id.__dict__) >= speaker:
                speaker_id = speaker
            else:
                raise ValueError(
                    f"Speaker id {speaker} >= number of speakers {len(self.spk2id.__dict__)}"
                )
        else:
            if speaker in self.spk2id.__dict__:
                speaker_id = self.spk2id.__dict__[speaker]
            else:
                LOG.warning(f"Speaker {speaker} is not found. Use speaker 0 instead.")
                speaker_id = 0
        sid = torch.LongTensor([int(speaker_id)]).to(self.dev).unsqueeze(0)

        # get unit f0
        c, f0, uv = self.get_unit_f0(audio, transpose, cluster_infer_ratio, speaker)
        if "half" in self.net_g_path and torch.cuda.is_available():
            c = c.half()

        # inference
        with torch.no_grad():
            with timer() as t:
                audio = self.net_g_ms.infer(
                    c,
                    f0=f0,
                    g=sid,
                    uv=uv,
                    predict_f0=auto_predict_f0,
                    noice_scale=noise_scale,
                )[0, 0].data.float()
            realtime_coef = len(audio) / (t.elapsed * self.target_sample)
            LOG.info(
                f"Inferece time: {t.elapsed:.2f}s, Realtime coef: {realtime_coef:.2f} "
                f"Input shape: {audio.shape}, Output shape: {audio.shape}"
            )
        return audio, audio.shape[-1]

    def clear_empty(self):
        torch.cuda.empty_cache()

    def infer_silence(
        self,
        audio: np.ndarray[Any, np.dtype[np.float32]],
        *,
        # svc config
        speaker: int | str,
        transpose: int = 0,
        auto_predict_f0: bool = False,
        cluster_infer_ratio: float = 0,
        noise_scale: float = 0.4,
        # slice config
        db_thresh: int = -40,
        pad_seconds: float = 0.5,
        fade_seconds: float = 0.04,
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        chunks = slicer.cut(audio, self.target_sample, db_thresh=db_thresh)
        LOG.info(f"Cut audio into chunks {chunks}")
        sr = self.target_sample

        result_audio = np.array([])
        for slice_tag, data in slicer.chunks2audio(audio, chunks):
            # segment length
            length = int(np.ceil(len(data) / sr * self.target_sample))
            if slice_tag:
                LOG.info("Skip silence")
                _audio = np.zeros(length)
            else:
                # pad
                pad_len = int(sr * pad_seconds)
                data = np.concatenate([np.zeros([pad_len]), data, np.zeros([pad_len])])
                out_audio, out_sr = self.infer(
                    speaker,
                    transpose,
                    audio,
                    cluster_infer_ratio=cluster_infer_ratio,
                    auto_predict_f0=auto_predict_f0,
                    noise_scale=noise_scale,
                )
                _audio = out_audio.cpu().numpy()
                pad_len = int(self.target_sample * pad_seconds)
                _audio = _audio[pad_len:-pad_len]

                # add fade
                fade_len = int(self.target_sample * fade_seconds)
                _audio[:fade_len] = _audio[:fade_len] * np.linspace(0, 1, fade_len)
                _audio[-fade_len:] = _audio[-fade_len:] * np.linspace(1, 0, fade_len)
            result_audio = np.concatenate([result_audio, pad_array(_audio, length)])
        result_audio = result_audio[: audio.shape[0]]
        return result_audio


import maad


class RealTimeVCBase:
    def __init__(
        self,
        *,
        svc_model: Svc,
        crossfade_len: int = 3840,
        use_slicer: bool = True,
    ):
        self.svc_model = svc_model
        self.crossfade_len = crossfade_len
        self.last_input = np.zeros(crossfade_len * 2, dtype=np.float32)
        self.last_infered = np.zeros(crossfade_len * 2, dtype=np.float32)
        self.use_slicer = use_slicer

    """The input and output are 1-dimensional numpy audio waveform arrays"""

    def process(
        self,
        input_audio: np.ndarray[Any, np.dtype[np.float32]],
        *,
        # svc config
        speaker: int | str,
        transpose: int,
        cluster_infer_ratio: float = 0,
        auto_predict_f0: bool = False,
        noise_scale: float = 0.4,
        # slice config
        db_thresh: int = -40,
        pad_seconds: float = 0.5,
    ):
        if input_audio.ndim != 1:
            raise ValueError("Input audio must be 1-dimensional.")
        if input_audio.shape[0] < self.crossfade_len:
            raise ValueError(
                f"Input audio length ({len(input_audio)}) should be at least crossfade length ({self.crossfade_len})."
            )
        input_audio = input_audio.astype(np.float32)
        input_audio = np.nan_to_num(input_audio)

        # create input audio
        input_audio_c = np.concatenate([self.last_input, input_audio])[
            -(input_audio.shape[0] + self.crossfade_len) :
        ]
        self.last_input = input_audio.copy()
        LOG.info(
            f"Input shape: {input_audio.shape}, Concatenated shape: {input_audio_c.shape}, Crossfade length: {self.crossfade_len}"
        )
        assert input_audio_c.shape[0] == input_audio.shape[0] + self.crossfade_len

        # infer
        if self.use_slicer:
            infered_audio_c = self.svc_model.infer_silence(
                audio=input_audio_c,
                speaker=speaker,
                transpose=transpose,
                cluster_infer_ratio=cluster_infer_ratio,
                auto_predict_f0=auto_predict_f0,
                noise_scale=noise_scale,
                db_thresh=db_thresh,
                pad_seconds=pad_seconds,
            )
        else:
            rms = np.sqrt(np.mean(input_audio**2))
            min_rms = 10 ** (db_thresh / 20)
            if rms < min_rms:
                LOG.info(f"Skip silence: RMS={rms:.2f} < {min_rms:.2f}")
                infered_audio_c = input_audio_c.copy()
            else:
                LOG.info(f"Start inference: RMS={rms:.2f} >= {min_rms:.2f}")
                infered_audio_c, _ = self.svc_model.infer(
                    speaker=speaker,
                    transpose=transpose,
                    audio=input_audio_c,
                    cluster_infer_ratio=cluster_infer_ratio,
                    auto_predict_f0=auto_predict_f0,
                    noise_scale=noise_scale,
                )
                infered_audio_c = infered_audio_c.cpu().numpy()
        infered_audio_c = infered_audio_c
        LOG.info(f"Concentrated Inferred shape: {infered_audio_c.shape}")
        assert infered_audio_c.shape[0] == input_audio_c.shape[0]

        # crossfade
        result = maad.util.crossfade(
            self.last_infered, infered_audio_c, 1, self.crossfade_len
        )[: input_audio.shape[0]]
        LOG.info(f"Result shape: {result.shape}")
        assert result.shape[0] == input_audio.shape[0]
        self.last_infered = infered_audio_c
        return result
