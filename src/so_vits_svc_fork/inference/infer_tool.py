from __future__ import annotations

import os
import queue
from logging import getLogger
from typing import Any, Iterable

import librosa
import numpy as np
import torch
from cm_time import timer
from numpy import dtype, float32, ndarray

from so_vits_svc_fork import cluster, utils
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


def split_silence2(
    audio: ndarray[Any, dtype[float32]],
    top_db: int = 40,
    ref: float = np.max,
    frame_length: int = 2048,
    hop_length: int = 512,
    aggregate: bool = True,
    pad_seconds: float = 0.5,
) -> Iterable[tuple[bool, ndarray[Any, dtype[float32]]]]:
    non_silence_indices = librosa.effects.split(
        audio, top_db=top_db, ref=ref, frame_length=frame_length, hop_length=hop_length
    )
    last_end = 0
    for start, end in non_silence_indices:
        if start - last_end > 0:
            yield False, audio[last_end:start]
        yield True, audio[start:end]
        last_end = end


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
    ) -> tuple[torch.Tensor, int]:
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
        # fade_seconds: float = 0.0,
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        sr = self.target_sample
        result_audio = np.array([])
        for slice_tag, data in split_silence2(audio):
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
                # fade_len = int(self.target_sample * fade_seconds)
                # _audio[:fade_len] = _audio[:fade_len] * np.linspace(0, 1, fade_len)
                # _audio[-fade_len:] = _audio[-fade_len:] * np.linspace(1, 0, fade_len)
            result_audio = np.concatenate([result_audio, pad_array(_audio, length)])
        result_audio = result_audio[: audio.shape[0]]
        return result_audio


class Crossfader:
    def __init__(self, *, crossfade_len: int) -> None:
        self.crossfade_len = crossfade_len
        self.last_input_left = np.zeros(crossfade_len, dtype=np.float32)
        self.last_infered_left = np.zeros(crossfade_len, dtype=np.float32)

    def process(
        self, input_audio: ndarray[Any, dtype[float32]], *args, **kwargs: Any
    ) -> ndarray[Any, dtype[float32]]:
        """
        chunks        : ■■■■■■□□□□□□
        add last input:□■■■■■■
                             ■□□□□□□
        infer         :□■■■■■■
                             ■□□□□□□
        crossfade     :▲■■■■■
                             ▲□□□□□
        """
        if input_audio.ndim != 1:
            raise ValueError("Input audio must be 1-dimensional.")
        if input_audio.shape[0] < self.crossfade_len:
            raise ValueError(
                f"Input audio length ({len(input_audio)}) should be at least crossfade length ({self.crossfade_len})."
            )
        input_audio = input_audio.astype(np.float32)
        input_audio_ = np.concatenate([self.last_input_left, input_audio])
        infer_audio_ = self.infer(input_audio_, *args, **kwargs)
        result_audio = np.concatenate(
            [
                (
                    self.last_infered_left * np.linspace(1, 0, self.crossfade_len)
                    + infer_audio_[: self.crossfade_len]
                    * np.linspace(0, 1, self.crossfade_len)
                )
                / 2,
                infer_audio_[self.crossfade_len : -self.crossfade_len],
            ]
        )
        self.last_input_left = input_audio[-self.crossfade_len :]
        self.last_infered_left = infer_audio_[-self.crossfade_len :]
        assert len(result_audio) == len(input_audio)
        return result_audio

    def infer(
        self, input_audio: ndarray[Any, dtype[float32]]
    ) -> ndarray[Any, dtype[float32]]:
        return input_audio


class RealtimeVC(Crossfader):
    def __init__(
        self,
        *,
        svc_model: Svc,
        crossfade_len: int = 3840,
        use_slicer: bool = True,
    ) -> None:
        self.svc_model = svc_model
        self.use_slicer = use_slicer
        super().__init__(crossfade_len=crossfade_len)

    def process(
        self,
        input_audio: ndarray[Any, dtype[float32]],
        *args: Any,
        **kwargs: Any,
    ) -> ndarray[Any, dtype[float32]]:
        return super().process(input_audio, *args, **kwargs)

    def infer(
        self,
        input_audio: np.ndarray[Any, np.dtype[np.float32]],
        # svc config
        speaker: int | str,
        transpose: int,
        cluster_infer_ratio: float = 0,
        auto_predict_f0: bool = False,
        noise_scale: float = 0.4,
        # slice config
        db_thresh: int = -40,
        pad_seconds: float = 0.5,
    ) -> ndarray[Any, dtype[float32]]:
        # infer
        if self.use_slicer:
            return self.svc_model.infer_silence(
                audio=input_audio,
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
                return input_audio.copy()
            else:
                LOG.info(f"Start inference: RMS={rms:.2f} >= {min_rms:.2f}")
                infered_audio_c, _ = self.svc_model.infer(
                    speaker=speaker,
                    transpose=transpose,
                    audio=input_audio,
                    cluster_infer_ratio=cluster_infer_ratio,
                    auto_predict_f0=auto_predict_f0,
                    noise_scale=noise_scale,
                )
                return infered_audio_c.cpu().numpy()


def split_silence(
    audio: ndarray[Any, dtype[float32]], resolution: int, db_thresh: int = -40
) -> Iterable[tuple[bool, ndarray[Any, dtype[float32]]]]:
    abs_ = np.abs(audio)
    abs_ma = np.array(
        [np.mean(chunk) for chunk in np.array_split(abs_, abs_.shape[0] // resolution)]
    )
    is_silence_array = abs_ma < 10 ** (db_thresh / 20)

    # yield
    is_silence_prev = None
    is_silence_changed = 0
    for i, is_silence in enumerate(is_silence_array):
        if is_silence != is_silence_prev:
            yield is_silence, audio[
                is_silence_changed
                * resolution : min(i * resolution, audio.shape[0] - 1)
            ]
            is_silence_changed = i
        is_silence_prev = is_silence


class RealtimeVC2:
    def __init__(self, svc_model: Svc, **kwargs) -> None:
        self.input_audio_store = np.array([], dtype=np.float32)
        self.output_queue = queue.Queue()
        self.svc_model = svc_model

    def process(
        self,
        input_audio: np.ndarray[Any, np.dtype[np.float32]],
        # svc config
        speaker: int | str,
        transpose: int,
        cluster_infer_ratio: float = 0,
        auto_predict_f0: bool = False,
        noise_scale: float = 0.4,
        # slice config
        db_thresh: int = -40,
        resolution_seconds: float = 0.1,
        **kwargs: Any,
    ) -> ndarray[Any, dtype[float32]]:
        def infer(audio: ndarray[Any, dtype[float32]]) -> ndarray[Any, dtype[float32]]:
            infered_audio_c, _ = self.svc_model.infer(
                speaker=speaker,
                transpose=transpose,
                audio=audio,
                cluster_infer_ratio=cluster_infer_ratio,
                auto_predict_f0=auto_predict_f0,
                noise_scale=noise_scale,
            )
            return infered_audio_c.cpu().numpy()

        self.input_audio_store = np.concatenate([self.input_audio_store, input_audio])
        LOG.debug(f"input_audio_store: {self.input_audio_store.shape}")
        non_silent_intervals = librosa.effects.split(
            self.input_audio_store, top_db=db_thresh
        )
        LOG.info(f"slice_silences: {non_silent_intervals}")
        assert len(non_silent_intervals) > 0
        if non_silent_intervals[-1][0] is False:
            # last is not silence
            self.input_audio_store = self.input_audio_store[
                -non_silent_intervals[-1][1].shape[0] :
            ]
        else:
            self.input_audio_store = np.array([], dtype=np.float32)

        last_silence_len = 0
        for i, (is_silence, slice_audio) in enumerate(non_silent_intervals):
            if i == len(non_silent_intervals) - 1:
                continue
            if not is_silence:
                slice_audio = infer(slice_audio)
                self.output_queue.put((last_silence_len, slice_audio))
                last_silence_len = 0
            else:
                last_silence_len += slice_audio.shape[0]
        LOG.info(f"Output queue: {self.output_queue.queue}")

        if self.output_queue.empty():
            return np.zeros_like(input_audio)
        total_output_queue_active_len = sum(
            [q[1].shape[0] for q in self.output_queue.queue]
        )
        input_audio_len = input_audio.shape[0]
        output_audio = np.array([], dtype=np.float32)
        if total_output_queue_active_len < input_audio_len:
            # not enough audio, use interpolation
            LOG.info("Not enough audio, use interpolation")
            total_silence_len = input_audio_len - total_output_queue_active_len
            total_output_queue_silence_len = sum(
                [q[0] for q in self.output_queue.queue]
            )
            silence_rate = total_silence_len / total_output_queue_silence_len

            for i, (silence_len, audio) in enumerate(self.output_queue.queue):
                output_audio = np.concatenate(
                    [
                        output_audio,
                        np.zeros(int(silence_len * silence_rate), dtype=np.float32),
                        audio,
                    ]
                )
            # clear queue
            self.output_queue = queue.Queue()
        else:
            # too much audio
            LOG.info("Too much audio, no silence")
            temp_len = 0
            audio = np.array([], dtype=np.float32)
            while temp_len < input_audio_len:
                silence_len, audio = self.output_queue.get()
                temp_len += silence_len
                output_audio = np.concatenate([output_audio, audio])
            else:
                self.output_queue.put(
                    (temp_len - input_audio_len, audio[-(temp_len - input_audio_len) :])
                )
        # fill
        output_audio = output_audio[:input_audio_len]
        output_audio = np.concatenate(
            [
                output_audio,
                np.repeat(output_audio[-1], output_audio.shape[0] - input_audio_len),
            ]
        )
        return output_audio
