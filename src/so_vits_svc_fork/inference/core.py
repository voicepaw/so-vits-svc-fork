from __future__ import annotations

from copy import deepcopy
from logging import getLogger
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

import attrs
import librosa
import numpy as np
import torch
from cm_time import timer
from numpy import dtype, float32, ndarray

import so_vits_svc_fork.f0
from so_vits_svc_fork import cluster, utils

from ..modules.synthesizers import SynthesizerTrn
from ..utils import get_optimal_device

LOG = getLogger(__name__)


def pad_array(array_, target_length: int):
    current_length = array_.shape[0]
    if current_length >= target_length:
        return array_[
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
            array_, (pad_left, pad_right), "constant", constant_values=(0, 0)
        )
        return padded_arr


@attrs.frozen(kw_only=True)
class Chunk:
    is_speech: bool
    audio: ndarray[Any, dtype[float32]]
    start: int
    end: int

    @property
    def duration(self) -> float32:
        # return self.end - self.start
        return float32(self.audio.shape[0])

    def __repr__(self) -> str:
        return f"Chunk(Speech: {self.is_speech}, {self.duration})"


def split_silence(
    audio: ndarray[Any, dtype[float32]],
    top_db: int = 40,
    ref: float | Callable[[ndarray[Any, dtype[float32]]], float] = 1,
    frame_length: int = 2048,
    hop_length: int = 512,
    aggregate: Callable[[ndarray[Any, dtype[float32]]], float] = np.mean,
    max_chunk_length: int = 0,
) -> Iterable[Chunk]:
    non_silence_indices = librosa.effects.split(
        audio,
        top_db=top_db,
        ref=ref,
        frame_length=frame_length,
        hop_length=hop_length,
        aggregate=aggregate,
    )
    last_end = 0
    for start, end in non_silence_indices:
        if start != last_end:
            yield Chunk(
                is_speech=False, audio=audio[last_end:start], start=last_end, end=start
            )
        while max_chunk_length > 0 and end - start > max_chunk_length:
            yield Chunk(
                is_speech=True,
                audio=audio[start : start + max_chunk_length],
                start=start,
                end=start + max_chunk_length,
            )
            start += max_chunk_length
        if end - start > 0:
            yield Chunk(is_speech=True, audio=audio[start:end], start=start, end=end)
        last_end = end
    if last_end != len(audio):
        yield Chunk(
            is_speech=False, audio=audio[last_end:], start=last_end, end=len(audio)
        )


class Svc:
    def __init__(
        self,
        *,
        net_g_path: Path | str,
        config_path: Path | str,
        device: torch.device | str | None = None,
        cluster_model_path: Path | str | None = None,
        half: bool = False,
    ):
        self.net_g_path = net_g_path
        if device is None:
            self.device = (get_optimal_device(),)
        else:
            self.device = torch.device(device)
        self.hps = utils.get_hparams(config_path)
        self.target_sample = self.hps.data.sampling_rate
        self.hop_size = self.hps.data.hop_length
        self.spk2id = self.hps.spk
        self.hubert_model = utils.get_hubert_model(
            self.device, self.hps.data.get("contentvec_final_proj", True)
        )
        self.dtype = torch.float16 if half else torch.float32
        self.contentvec_final_proj = self.hps.data.__dict__.get(
            "contentvec_final_proj", True
        )
        self.load_model()
        if cluster_model_path is not None and Path(cluster_model_path).exists():
            self.cluster_model = cluster.get_cluster_model(cluster_model_path)

    def load_model(self):
        self.net_g = SynthesizerTrn(
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            **self.hps.model,
        )
        _ = utils.load_checkpoint(self.net_g_path, self.net_g, None)
        _ = self.net_g.eval()
        for m in self.net_g.modules():
            utils.remove_weight_norm_if_exists(m)
        _ = self.net_g.to(self.device, dtype=self.dtype)
        self.net_g = self.net_g

    def get_unit_f0(
        self,
        audio: ndarray[Any, dtype[float32]],
        tran: int,
        cluster_infer_ratio: float,
        speaker: int | str,
        f0_method: Literal[
            "crepe", "crepe-tiny", "parselmouth", "dio", "harvest"
        ] = "dio",
    ):
        f0 = so_vits_svc_fork.f0.compute_f0(
            audio,
            sampling_rate=self.target_sample,
            hop_length=self.hop_size,
            method=f0_method,
        )
        f0, uv = so_vits_svc_fork.f0.interpolate_f0(f0)
        f0 = torch.as_tensor(f0, dtype=self.dtype, device=self.device)
        uv = torch.as_tensor(uv, dtype=self.dtype, device=self.device)
        f0 = f0 * 2 ** (tran / 12)
        f0 = f0.unsqueeze(0)
        uv = uv.unsqueeze(0)

        c = utils.get_content(
            self.hubert_model,
            audio,
            self.device,
            self.target_sample,
            self.contentvec_final_proj,
        ).to(self.dtype)
        c = utils.repeat_expand_2d(c.squeeze(0), f0.shape[1])

        if cluster_infer_ratio != 0:
            cluster_c = cluster.get_cluster_center_result(
                self.cluster_model, c.cpu().numpy().T, speaker
            ).T
            cluster_c = torch.FloatTensor(cluster_c).to(self.device)
            c = cluster_infer_ratio * cluster_c + (1 - cluster_infer_ratio) * c

        c = c.unsqueeze(0)
        return c, f0, uv

    def infer(
        self,
        speaker: int | str,
        transpose: int,
        audio: ndarray[Any, dtype[float32]],
        cluster_infer_ratio: float = 0,
        auto_predict_f0: bool = False,
        noise_scale: float = 0.4,
        f0_method: Literal[
            "crepe", "crepe-tiny", "parselmouth", "dio", "harvest"
        ] = "dio",
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
        speaker_candidates = list(
            filter(lambda x: x[1] == speaker_id, self.spk2id.__dict__.items())
        )
        if len(speaker_candidates) > 1:
            raise ValueError(
                f"Speaker_id {speaker_id} is not unique. Candidates: {speaker_candidates}"
            )
        elif len(speaker_candidates) == 0:
            raise ValueError(f"Speaker_id {speaker_id} is not found.")
        speaker = speaker_candidates[0][0]
        sid = torch.LongTensor([int(speaker_id)]).to(self.device).unsqueeze(0)

        # get unit f0
        c, f0, uv = self.get_unit_f0(
            audio, transpose, cluster_infer_ratio, speaker, f0_method
        )

        # inference
        with torch.no_grad():
            with timer() as t:
                audio = self.net_g.infer(
                    c,
                    f0=f0,
                    g=sid,
                    uv=uv,
                    predict_f0=auto_predict_f0,
                    noice_scale=noise_scale,
                )[0, 0].data.float()
            audio_duration = audio.shape[-1] / self.target_sample
            LOG.info(
                f"Inference time: {t.elapsed:.2f}s, RTF: {t.elapsed / audio_duration:.2f}"
            )
        torch.cuda.empty_cache()
        return audio, audio.shape[-1]

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
        f0_method: Literal[
            "crepe", "crepe-tiny", "parselmouth", "dio", "harvest"
        ] = "dio",
        # slice config
        db_thresh: int = -40,
        pad_seconds: float = 0.5,
        chunk_seconds: float = 0.5,
        absolute_thresh: bool = False,
        max_chunk_seconds: float = 40,
        # fade_seconds: float = 0.0,
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        sr = self.target_sample
        result_audio = np.array([], dtype=np.float32)
        chunk_length_min = chunk_length_min = (
            int(
                min(
                    sr / so_vits_svc_fork.f0.f0_min * 20 + 1,
                    chunk_seconds * sr,
                )
            )
            // 2
        )
        for chunk in split_silence(
            audio,
            top_db=-db_thresh,
            frame_length=chunk_length_min * 2,
            hop_length=chunk_length_min,
            ref=1 if absolute_thresh else np.max,
            max_chunk_length=int(max_chunk_seconds * sr),
        ):
            LOG.info(f"Chunk: {chunk}")
            if not chunk.is_speech:
                audio_chunk_infer = np.zeros_like(chunk.audio)
            else:
                # pad
                pad_len = int(sr * pad_seconds)
                audio_chunk_pad = np.concatenate(
                    [
                        np.zeros([pad_len], dtype=np.float32),
                        chunk.audio,
                        np.zeros([pad_len], dtype=np.float32),
                    ]
                )
                audio_chunk_pad_infer_tensor, _ = self.infer(
                    speaker,
                    transpose,
                    audio_chunk_pad,
                    cluster_infer_ratio=cluster_infer_ratio,
                    auto_predict_f0=auto_predict_f0,
                    noise_scale=noise_scale,
                    f0_method=f0_method,
                )
                audio_chunk_pad_infer = audio_chunk_pad_infer_tensor.cpu().numpy()
                pad_len = int(self.target_sample * pad_seconds)
                cut_len_2 = (len(audio_chunk_pad_infer) - len(chunk.audio)) // 2
                audio_chunk_infer = audio_chunk_pad_infer[
                    cut_len_2 : cut_len_2 + len(chunk.audio)
                ]

                # add fade
                # fade_len = int(self.target_sample * fade_seconds)
                # _audio[:fade_len] = _audio[:fade_len] * np.linspace(0, 1, fade_len)
                # _audio[-fade_len:] = _audio[-fade_len:] * np.linspace(1, 0, fade_len)

                # empty cache
                torch.cuda.empty_cache()
            result_audio = np.concatenate([result_audio, audio_chunk_infer])
        result_audio = result_audio[: audio.shape[0]]
        return result_audio


def sola_crossfade(
    first: ndarray[Any, dtype[float32]],
    second: ndarray[Any, dtype[float32]],
    crossfade_len: int,
    sola_search_len: int,
) -> ndarray[Any, dtype[float32]]:
    cor_nom = np.convolve(
        second[: sola_search_len + crossfade_len],
        np.flip(first[-crossfade_len:]),
        "valid",
    )
    cor_den = np.sqrt(
        np.convolve(
            second[: sola_search_len + crossfade_len] ** 2,
            np.ones(crossfade_len),
            "valid",
        )
        + 1e-8
    )
    sola_shift = np.argmax(cor_nom / cor_den)
    LOG.info(f"SOLA shift: {sola_shift}")
    second = second[sola_shift : sola_shift + len(second) - sola_search_len]
    return np.concatenate(
        [
            first[:-crossfade_len],
            first[-crossfade_len:] * np.linspace(1, 0, crossfade_len)
            + second[:crossfade_len] * np.linspace(0, 1, crossfade_len),
            second[crossfade_len:],
        ]
    )


class Crossfader:
    def __init__(
        self,
        *,
        additional_infer_before_len: int,
        additional_infer_after_len: int,
        crossfade_len: int,
        sola_search_len: int = 384,
    ) -> None:
        if additional_infer_before_len < 0:
            raise ValueError("additional_infer_len must be >= 0")
        if crossfade_len < 0:
            raise ValueError("crossfade_len must be >= 0")
        if additional_infer_after_len < 0:
            raise ValueError("additional_infer_len must be >= 0")
        if additional_infer_before_len < 0:
            raise ValueError("additional_infer_len must be >= 0")
        self.additional_infer_before_len = additional_infer_before_len
        self.additional_infer_after_len = additional_infer_after_len
        self.crossfade_len = crossfade_len
        self.sola_search_len = sola_search_len
        self.last_input_left = np.zeros(
            sola_search_len
            + crossfade_len
            + additional_infer_before_len
            + additional_infer_after_len,
            dtype=np.float32,
        )
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
        # check input
        if input_audio.ndim != 1:
            raise ValueError("Input audio must be 1-dimensional.")
        if (
            input_audio.shape[0] + self.additional_infer_before_len
            <= self.crossfade_len
        ):
            raise ValueError(
                f"Input audio length ({input_audio.shape[0]}) + additional_infer_len ({self.additional_infer_before_len}) must be greater than crossfade_len ({self.crossfade_len})."
            )
        input_audio = input_audio.astype(np.float32)
        input_audio_len = len(input_audio)

        # concat last input and infer
        input_audio_concat = np.concatenate([self.last_input_left, input_audio])
        del input_audio
        pad_len = 0
        if pad_len:
            infer_audio_concat = self.infer(
                np.pad(input_audio_concat, (pad_len, pad_len), mode="reflect"),
                *args,
                **kwargs,
            )[pad_len:-pad_len]
        else:
            infer_audio_concat = self.infer(input_audio_concat, *args, **kwargs)

        # debug SOLA (using copy synthesis with a random shift)
        """
        rs = int(np.random.uniform(-200,200))
        LOG.info(f"Debug random shift: {rs}")
        infer_audio_concat = np.roll(input_audio_concat, rs)
        """

        if len(infer_audio_concat) != len(input_audio_concat):
            raise ValueError(
                f"Inferred audio length ({len(infer_audio_concat)}) should be equal to input audio length ({len(input_audio_concat)})."
            )
        infer_audio_to_use = infer_audio_concat[
            -(
                self.sola_search_len
                + self.crossfade_len
                + input_audio_len
                + self.additional_infer_after_len
            ) : -self.additional_infer_after_len
        ]
        assert (
            len(infer_audio_to_use)
            == input_audio_len + self.sola_search_len + self.crossfade_len
        ), f"{len(infer_audio_to_use)} != {input_audio_len + self.sola_search_len + self.cross_fade_len}"
        _audio = sola_crossfade(
            self.last_infered_left,
            infer_audio_to_use,
            self.crossfade_len,
            self.sola_search_len,
        )
        result_audio = _audio[: -self.crossfade_len]
        assert (
            len(result_audio) == input_audio_len
        ), f"{len(result_audio)} != {input_audio_len}"

        # update last input and inferred
        self.last_input_left = input_audio_concat[
            -(
                self.sola_search_len
                + self.crossfade_len
                + self.additional_infer_before_len
                + self.additional_infer_after_len
            ) :
        ]
        self.last_infered_left = _audio[-self.crossfade_len :]
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
        additional_infer_before_len: int = 7680,
        additional_infer_after_len: int = 7680,
        split: bool = True,
    ) -> None:
        self.svc_model = svc_model
        self.split = split
        super().__init__(
            crossfade_len=crossfade_len,
            additional_infer_before_len=additional_infer_before_len,
            additional_infer_after_len=additional_infer_after_len,
        )

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
        f0_method: Literal[
            "crepe", "crepe-tiny", "parselmouth", "dio", "harvest"
        ] = "dio",
        # slice config
        db_thresh: int = -40,
        pad_seconds: float = 0.5,
        chunk_seconds: float = 0.5,
    ) -> ndarray[Any, dtype[float32]]:
        # infer
        if self.split:
            return self.svc_model.infer_silence(
                audio=input_audio,
                speaker=speaker,
                transpose=transpose,
                cluster_infer_ratio=cluster_infer_ratio,
                auto_predict_f0=auto_predict_f0,
                noise_scale=noise_scale,
                f0_method=f0_method,
                db_thresh=db_thresh,
                pad_seconds=pad_seconds,
                chunk_seconds=chunk_seconds,
                absolute_thresh=True,
            )
        else:
            rms = np.sqrt(np.mean(input_audio**2))
            min_rms = 10 ** (db_thresh / 20)
            if rms < min_rms:
                LOG.info(f"Skip silence: RMS={rms:.2f} < {min_rms:.2f}")
                return np.zeros_like(input_audio)
            else:
                LOG.info(f"Start inference: RMS={rms:.2f} >= {min_rms:.2f}")
                infered_audio_c, _ = self.svc_model.infer(
                    speaker=speaker,
                    transpose=transpose,
                    audio=input_audio,
                    cluster_infer_ratio=cluster_infer_ratio,
                    auto_predict_f0=auto_predict_f0,
                    noise_scale=noise_scale,
                    f0_method=f0_method,
                )
                return infered_audio_c.cpu().numpy()


class RealtimeVC2:
    chunk_store: list[Chunk]

    def __init__(self, svc_model: Svc) -> None:
        self.input_audio_store = np.array([], dtype=np.float32)
        self.chunk_store = []
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
        f0_method: Literal[
            "crepe", "crepe-tiny", "parselmouth", "dio", "harvest"
        ] = "dio",
        # slice config
        db_thresh: int = -40,
        chunk_seconds: float = 0.5,
    ) -> ndarray[Any, dtype[float32]]:
        def infer(audio: ndarray[Any, dtype[float32]]) -> ndarray[Any, dtype[float32]]:
            infered_audio_c, _ = self.svc_model.infer(
                speaker=speaker,
                transpose=transpose,
                audio=audio,
                cluster_infer_ratio=cluster_infer_ratio,
                auto_predict_f0=auto_predict_f0,
                noise_scale=noise_scale,
                f0_method=f0_method,
            )
            return infered_audio_c.cpu().numpy()

        self.input_audio_store = np.concatenate([self.input_audio_store, input_audio])
        LOG.info(f"input_audio_store: {self.input_audio_store.shape}")
        sr = self.svc_model.target_sample
        chunk_length_min = (
            int(min(sr / so_vits_svc_fork.f0.f0_min * 20 + 1, chunk_seconds * sr)) // 2
        )
        LOG.info(f"Chunk length min: {chunk_length_min}")
        chunk_list = list(
            split_silence(
                self.input_audio_store,
                -db_thresh,
                frame_length=chunk_length_min * 2,
                hop_length=chunk_length_min,
                ref=1,  # use absolute threshold
            )
        )
        assert len(chunk_list) > 0
        LOG.info(f"Chunk list: {chunk_list}")
        # do not infer LAST incomplete is_speech chunk and save to store
        if chunk_list[-1].is_speech:
            self.input_audio_store = chunk_list.pop().audio
        else:
            self.input_audio_store = np.array([], dtype=np.float32)

        # infer complete is_speech chunk and save to store
        self.chunk_store.extend(
            [
                attrs.evolve(c, audio=infer(c.audio) if c.is_speech else c.audio)
                for c in chunk_list
            ]
        )

        # calculate lengths and determine compress rate
        total_speech_len = sum(
            [c.duration if c.is_speech else 0 for c in self.chunk_store]
        )
        total_silence_len = sum(
            [c.duration if not c.is_speech else 0 for c in self.chunk_store]
        )
        input_audio_len = input_audio.shape[0]
        silence_compress_rate = total_silence_len / max(
            0, input_audio_len - total_speech_len
        )
        LOG.info(
            f"Total speech len: {total_speech_len}, silence len: {total_silence_len}, silence compress rate: {silence_compress_rate}"
        )

        # generate output audio
        output_audio = np.array([], dtype=np.float32)
        break_flag = False
        LOG.info(f"Chunk store: {self.chunk_store}")
        for chunk in deepcopy(self.chunk_store):
            compress_rate = 1 if chunk.is_speech else silence_compress_rate
            left_len = input_audio_len - output_audio.shape[0]
            # calculate chunk duration
            chunk_duration_output = int(min(chunk.duration / compress_rate, left_len))
            chunk_duration_input = int(min(chunk.duration, left_len * compress_rate))
            LOG.info(
                f"Chunk duration output: {chunk_duration_output}, input: {chunk_duration_input}, left len: {left_len}"
            )

            # remove chunk from store
            self.chunk_store.pop(0)
            if chunk.duration > chunk_duration_input:
                left_chunk = attrs.evolve(
                    chunk, audio=chunk.audio[chunk_duration_input:]
                )
                chunk = attrs.evolve(chunk, audio=chunk.audio[:chunk_duration_input])

                self.chunk_store.insert(0, left_chunk)
                break_flag = True

            if chunk.is_speech:
                # if is_speech, just concat
                output_audio = np.concatenate([output_audio, chunk.audio])
            else:
                # if is_silence, concat with zeros and compress with silence_compress_rate
                output_audio = np.concatenate(
                    [
                        output_audio,
                        np.zeros(
                            chunk_duration_output,
                            dtype=np.float32,
                        ),
                    ]
                )

            if break_flag:
                break
        LOG.info(f"Chunk store: {self.chunk_store}, output_audio: {output_audio.shape}")
        # make same length (errors)
        output_audio = output_audio[:input_audio_len]
        output_audio = np.concatenate(
            [
                output_audio,
                np.zeros(input_audio_len - output_audio.shape[0], dtype=np.float32),
            ]
        )
        return output_audio
