from __future__ import annotations

import io
from logging import getLogger
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile
import torch
from tqdm import tqdm

from .inference import infer_tool, slicer
from .inference.infer_tool import Svc

LOG = getLogger(__name__)


def infer(
    input_path: Path,
    output_path: Path,
    speaker: str,
    model_path: Path,
    config_path: Path,
    cluster_model_path: "Path | None" = None,
    transpose: int = 0,
    db_thresh: int = -40,
    auto_predict_f0: bool = False,
    cluster_infer_ratio: float = 0,
    noice_scale: float = 0.4,
    pad_seconds: float = 0.5,
    device: Literal["cpu", "cuda"] = "cuda" if torch.cuda.is_available() else "cpu",
):
    svc_model = Svc(model_path.as_posix(), config_path.as_posix(), cluster_model_path, device)
    # infer_tool.fill_a_to_b(transpose, input_path)

    raw_audio_path = input_path
    infer_tool.format_wav(raw_audio_path)
    wav_path = Path(raw_audio_path).with_suffix(".wav")
    chunks = slicer.cut(wav_path, db_thresh=db_thresh)
    audio_data, audio_sr = slicer.chunks2audio(wav_path, chunks)

    audio = []
    for slice_tag, data in tqdm(audio_data):
        # segment length
        length = int(np.ceil(len(data) / audio_sr * svc_model.target_sample))
        if slice_tag:
            LOG.info("skip non-speaking segment")
            _audio = np.zeros(length)
        else:
            # pad
            pad_len = int(audio_sr * pad_seconds)
            data = np.concatenate([np.zeros([pad_len]), data, np.zeros([pad_len])])
            raw_path = io.BytesIO()
            soundfile.write(raw_path, data, audio_sr, format="wav")
            raw_path.seek(0)
            out_audio, out_sr = svc_model.infer(
                speaker,
                transpose,
                raw_path,
                cluster_infer_ratio=cluster_infer_ratio,
                auto_predict_f0=auto_predict_f0,
                noice_scale=noice_scale,
            )
            _audio = out_audio.cpu().numpy()
            pad_len = int(svc_model.target_sample * pad_seconds)
            _audio = _audio[pad_len:-pad_len]

        audio.extend(list(infer_tool.pad_array(_audio, length)))

    soundfile.write(output_path, audio, svc_model.target_sample)
