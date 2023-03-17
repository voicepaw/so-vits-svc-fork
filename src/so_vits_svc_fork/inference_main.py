from __future__ import annotations

from logging import getLogger
from pathlib import Path
from typing import Literal

import librosa
import numpy as np
import soundfile
import torch

from .inference.infer_tool import RealTimeVCBase, Svc

LOG = getLogger(__name__)


def infer(
    *,
    # paths
    input_path: Path,
    output_path: Path,
    model_path: Path,
    config_path: Path,
    # svc config
    speaker: str,
    cluster_model_path: Path | None = None,
    transpose: int = 0,
    auto_predict_f0: bool = False,
    cluster_infer_ratio: float = 0,
    noise_scale: float = 0.4,
    # slice config
    db_thresh: int = -40,
    pad_seconds: float = 0.5,
    device: Literal["cpu", "cuda"] = "cuda" if torch.cuda.is_available() else "cpu",
):
    svc_model = Svc(
        net_g_path=model_path.as_posix(),
        config_path=config_path.as_posix(),
        cluster_model_path=cluster_model_path.as_posix()
        if cluster_model_path
        else None,
        device=device,
    )

    wav, sr = librosa.load(input_path, sr=svc_model.target_sample)
    audio = svc_model.infer_silence(
        wav,
        speaker=speaker,
        db_thresh=db_thresh,
        pad_seconds=pad_seconds,
        transpose=transpose,
        auto_predict_f0=auto_predict_f0,
        cluster_infer_ratio=cluster_infer_ratio,
        noise_scale=noise_scale,
    )

    soundfile.write(output_path, audio, svc_model.target_sample)


import sounddevice as sd


def realtime(
    *,
    # paths
    model_path: Path,
    config_path: Path,
    # svc config
    speaker: str,
    cluster_model_path: Path | None = None,
    transpose: int = 0,
    auto_predict_f0: bool = False,
    cluster_infer_ratio: float = 0,
    noise_scale: float = 0.4,
    # slice config
    db_thresh: int = -40,
    pad_seconds: float = 0.5,
    # realtime config
    crossfade_seconds: float = 0.05,
    block_seconds: float = 0.5,
    device: Literal["cpu", "cuda"] = "cuda" if torch.cuda.is_available() else "cpu",
):
    svc_model = Svc(
        net_g_path=model_path.as_posix(),
        config_path=config_path.as_posix(),
        cluster_model_path=cluster_model_path.as_posix()
        if cluster_model_path
        else None,
        device=device,
    )
    model = RealTimeVCBase(
        svc_model=svc_model,
        crossfade_len=int(crossfade_seconds * svc_model.target_sample),
    )

    def callback(
        indata: np.ndarray,
        outdata: np.ndarray,
        frames: int,
        time: int,
        status: sd.CallbackFlags,
    ) -> None:
        LOG.info(f"Frames: {frames}, Status: {status}, Shape: {indata.shape}")

        outdata[:] = model.process(
            input_audio=indata.mean(axis=1),
            speaker=speaker,
            transpose=transpose,
            auto_predict_f0=auto_predict_f0,
            noise_scale=noise_scale,
            cluster_infer_ratio=cluster_infer_ratio,
            db_thresh=db_thresh,
            pad_seconds=pad_seconds,
        ).reshape(-1, 1)

    with sd.Stream(
        channels=1,
        callback=callback,
        samplerate=svc_model.target_sample,
        blocksize=int(block_seconds * svc_model.target_sample),
    ):
        while True:
            sd.sleep(1)
