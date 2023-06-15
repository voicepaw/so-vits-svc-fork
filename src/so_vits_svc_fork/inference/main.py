from __future__ import annotations

from logging import getLogger
from pathlib import Path
from typing import Literal, Sequence

import librosa
import numpy as np
import soundfile
import torch
from cm_time import timer
from tqdm import tqdm

from so_vits_svc_fork.inference.core import RealtimeVC, RealtimeVC2, Svc
from so_vits_svc_fork.utils import get_optimal_device

LOG = getLogger(__name__)


def infer(
    *,
    # paths
    input_path: Path | str | Sequence[Path | str],
    output_path: Path | str | Sequence[Path | str],
    model_path: Path | str,
    config_path: Path | str,
    recursive: bool = False,
    # svc config
    speaker: int | str,
    cluster_model_path: Path | str | None = None,
    transpose: int = 0,
    auto_predict_f0: bool = False,
    cluster_infer_ratio: float = 0,
    noise_scale: float = 0.4,
    f0_method: Literal["crepe", "crepe-tiny", "parselmouth", "dio", "harvest"] = "dio",
    # slice config
    db_thresh: int = -40,
    pad_seconds: float = 0.5,
    chunk_seconds: float = 0.5,
    absolute_thresh: bool = False,
    max_chunk_seconds: float = 40,
    device: str | torch.device = get_optimal_device(),
):
    if isinstance(input_path, (str, Path)):
        input_path = [input_path]
    if isinstance(output_path, (str, Path)):
        output_path = [output_path]
    if len(input_path) != len(output_path):
        raise ValueError(
            f"input_path and output_path must have same length, but got {len(input_path)} and {len(output_path)}"
        )

    model_path = Path(model_path)
    config_path = Path(config_path)
    output_path = [Path(p) for p in output_path]
    input_path = [Path(p) for p in input_path]
    output_paths = []
    input_paths = []

    for input_path, output_path in zip(input_path, output_path):
        if input_path.is_dir():
            if not recursive:
                raise ValueError(
                    f"input_path is a directory, but recursive is False: {input_path}"
                )
            input_paths.extend(list(input_path.rglob("*.*")))
            output_paths.extend(
                [output_path / p.relative_to(input_path) for p in input_paths]
            )
            continue
        input_paths.append(input_path)
        output_paths.append(output_path)

    cluster_model_path = Path(cluster_model_path) if cluster_model_path else None
    svc_model = Svc(
        net_g_path=model_path.as_posix(),
        config_path=config_path.as_posix(),
        cluster_model_path=cluster_model_path.as_posix()
        if cluster_model_path
        else None,
        device=device,
    )

    try:
        pbar = tqdm(list(zip(input_paths, output_paths)), disable=len(input_paths) == 1)
        for input_path, output_path in pbar:
            pbar.set_description(f"{input_path}")
            try:
                audio, _ = librosa.load(str(input_path), sr=svc_model.target_sample)
            except Exception as e:
                LOG.error(f"Failed to load {input_path}")
                LOG.exception(e)
                continue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            audio = svc_model.infer_silence(
                audio.astype(np.float32),
                speaker=speaker,
                transpose=transpose,
                auto_predict_f0=auto_predict_f0,
                cluster_infer_ratio=cluster_infer_ratio,
                noise_scale=noise_scale,
                f0_method=f0_method,
                db_thresh=db_thresh,
                pad_seconds=pad_seconds,
                chunk_seconds=chunk_seconds,
                absolute_thresh=absolute_thresh,
                max_chunk_seconds=max_chunk_seconds,
            )
            soundfile.write(str(output_path), audio, svc_model.target_sample)
    finally:
        del svc_model
        torch.cuda.empty_cache()


def realtime(
    *,
    # paths
    model_path: Path | str,
    config_path: Path | str,
    # svc config
    speaker: str,
    cluster_model_path: Path | str | None = None,
    transpose: int = 0,
    auto_predict_f0: bool = False,
    cluster_infer_ratio: float = 0,
    noise_scale: float = 0.4,
    f0_method: Literal["crepe", "crepe-tiny", "parselmouth", "dio", "harvest"] = "dio",
    # slice config
    db_thresh: int = -40,
    pad_seconds: float = 0.5,
    chunk_seconds: float = 0.5,
    # realtime config
    crossfade_seconds: float = 0.05,
    additional_infer_before_seconds: float = 0.2,
    additional_infer_after_seconds: float = 0.1,
    block_seconds: float = 0.5,
    version: int = 2,
    input_device: int | str | None = None,
    output_device: int | str | None = None,
    device: str | torch.device = get_optimal_device(),
    passthrough_original: bool = False,
):
    import sounddevice as sd

    model_path = Path(model_path)
    config_path = Path(config_path)
    cluster_model_path = Path(cluster_model_path) if cluster_model_path else None
    svc_model = Svc(
        net_g_path=model_path.as_posix(),
        config_path=config_path.as_posix(),
        cluster_model_path=cluster_model_path.as_posix()
        if cluster_model_path
        else None,
        device=device,
    )

    LOG.info("Creating realtime model...")
    if version == 1:
        model = RealtimeVC(
            svc_model=svc_model,
            crossfade_len=int(crossfade_seconds * svc_model.target_sample),
            additional_infer_before_len=int(
                additional_infer_before_seconds * svc_model.target_sample
            ),
            additional_infer_after_len=int(
                additional_infer_after_seconds * svc_model.target_sample
            ),
        )
    else:
        model = RealtimeVC2(
            svc_model=svc_model,
        )

    # LOG all device info
    devices = sd.query_devices()
    LOG.info(f"Device: {devices}")
    if isinstance(input_device, str):
        input_device_candidates = [
            i for i, d in enumerate(devices) if d["name"] == input_device
        ]
        if len(input_device_candidates) == 0:
            LOG.warning(f"Input device {input_device} not found, using default")
            input_device = None
        else:
            input_device = input_device_candidates[0]
    if isinstance(output_device, str):
        output_device_candidates = [
            i for i, d in enumerate(devices) if d["name"] == output_device
        ]
        if len(output_device_candidates) == 0:
            LOG.warning(f"Output device {output_device} not found, using default")
            output_device = None
        else:
            output_device = output_device_candidates[0]
    if input_device is None or input_device >= len(devices):
        input_device = sd.default.device[0]
    if output_device is None or output_device >= len(devices):
        output_device = sd.default.device[1]
    LOG.info(
        f"Input Device: {devices[input_device]['name']}, Output Device: {devices[output_device]['name']}"
    )

    # the model RTL is somewhat significantly high only in the first inference
    # there could be no better way to warm up the model than to do a dummy inference
    # (there are not differences in the behavior of the model between the first and the later inferences)
    # so we do a dummy inference to warm up the model (1 second of audio)
    LOG.info("Warming up the model...")
    svc_model.infer(
        speaker=speaker,
        transpose=transpose,
        auto_predict_f0=auto_predict_f0,
        cluster_infer_ratio=cluster_infer_ratio,
        noise_scale=noise_scale,
        f0_method=f0_method,
        audio=np.zeros(svc_model.target_sample, dtype=np.float32),
    )

    def callback(
        indata: np.ndarray,
        outdata: np.ndarray,
        frames: int,
        time: int,
        status: sd.CallbackFlags,
    ) -> None:
        LOG.debug(
            f"Frames: {frames}, Status: {status}, Shape: {indata.shape}, Time: {time}"
        )

        kwargs = dict(
            input_audio=indata.mean(axis=1).astype(np.float32),
            # svc config
            speaker=speaker,
            transpose=transpose,
            auto_predict_f0=auto_predict_f0,
            cluster_infer_ratio=cluster_infer_ratio,
            noise_scale=noise_scale,
            f0_method=f0_method,
            # slice config
            db_thresh=db_thresh,
            # pad_seconds=pad_seconds,
            chunk_seconds=chunk_seconds,
        )
        if version == 1:
            kwargs["pad_seconds"] = pad_seconds
        with timer() as t:
            inference = model.process(
                **kwargs,
            ).reshape(-1, 1)
        if passthrough_original:
            outdata[:] = (indata + inference) / 2
        else:
            outdata[:] = inference
        rtf = t.elapsed / block_seconds
        LOG.info(f"Realtime inference time: {t.elapsed:.3f}s, RTF: {rtf:.3f}")
        if rtf > 1:
            LOG.warning("RTF is too high, consider increasing block_seconds")

    try:
        with sd.Stream(
            device=(input_device, output_device),
            channels=1,
            callback=callback,
            samplerate=svc_model.target_sample,
            blocksize=int(block_seconds * svc_model.target_sample),
            latency="low",
        ) as stream:
            LOG.info(f"Latency: {stream.latency}")
            while True:
                sd.sleep(1000)
    finally:
        # del model, svc_model
        torch.cuda.empty_cache()
