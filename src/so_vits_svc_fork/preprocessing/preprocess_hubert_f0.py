from __future__ import annotations

from logging import getLogger
from pathlib import Path
from random import shuffle
from typing import Iterable, Literal

import librosa
import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

import so_vits_svc_fork.f0
from so_vits_svc_fork import utils

from ..utils import get_total_gpu_memory
from .preprocess_utils import check_hubert_min_duration

LOG = getLogger(__name__)
HUBERT_MEMORY = 1600
HUBERT_MEMORY_CREPE = 2600


def _process_one(
    *,
    filepath: Path,
    hubert_model,
    sampling_rate: int,
    hop_length: int,
    device: Literal["cuda", "cpu"] = "cuda",
    f0_method: Literal["crepe", "crepe-tiny", "parselmouth", "dio", "harvest"] = "dio",
    force_rebuild: bool = False,
    legacy_final_proj: bool = False,
):
    audio, sr = librosa.load(filepath, sr=sampling_rate)

    if not check_hubert_min_duration(audio, sr):
        LOG.info(f"Skip {filepath} because it is too short.")
        return

    # Compute HuBERT content
    soft_path = filepath.parent / (filepath.name + ".soft.pt")
    if (not soft_path.exists()) or force_rebuild:
        c = utils.get_content(
            hubert_model, audio, device, sr=sr, legacy_final_proj=legacy_final_proj
        )
        torch.save(c.cpu(), soft_path)
    else:
        LOG.info(f"Skip {filepath} because {soft_path} exists.")

    # Compute f0
    f0_path = filepath.parent / (filepath.name + ".f0.npy")
    if (not f0_path.exists()) or force_rebuild:
        f0 = so_vits_svc_fork.f0.compute_f0(
            audio, sampling_rate=sampling_rate, hop_length=hop_length, method=f0_method
        )
        np.save(f0_path, f0)
    else:
        LOG.info(f"Skip {filepath} because {f0_path} exists.")
    torch.cuda.empty_cache()


def _process_batch(
    *,
    filepaths: Iterable[Path],
    sampling_rate: int,
    hop_length: int,
    pbar_position: int,
    f0_method: Literal["crepe", "crepe-tiny", "parselmouth", "dio", "harvest"] = "dio",
    force_rebuild: bool = False,
    legacy_final_proj: bool = False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hubert_model = utils.get_hubert_model(device)

    for filepath in tqdm(filepaths, position=pbar_position):
        _process_one(
            filepath=filepath,
            hubert_model=hubert_model,
            sampling_rate=sampling_rate,
            hop_length=hop_length,
            device=device,
            f0_method=f0_method,
            force_rebuild=force_rebuild,
            legacy_final_proj=legacy_final_proj,
        )


def preprocess_hubert_f0(
    input_dir: Path | str,
    config_path: Path | str,
    n_jobs: int | None = None,
    f0_method: Literal["crepe", "crepe-tiny", "parselmouth", "dio", "harvest"] = "dio",
    force_rebuild: bool = False,
):
    input_dir = Path(input_dir)
    config_path = Path(config_path)
    utils.ensure_pretrained_model(".", "contentvec")
    hps = utils.get_hparams(config_path)
    if n_jobs is None:
        memory = get_total_gpu_memory("free")
        n_jobs = memory // (
            HUBERT_MEMORY_CREPE if f0_method == "crepe" else HUBERT_MEMORY
        )
        LOG.info(f"n_jobs automatically set to {n_jobs}, memory: {memory} MiB")

    filepaths = list(input_dir.rglob("*.wav"))
    n_jobs = min(len(filepaths) // 16 + 1, n_jobs)
    shuffle(filepaths)
    filepath_chunks = np.array_split(filepaths, n_jobs)
    Parallel(n_jobs=n_jobs)(
        delayed(_process_batch)(
            filepaths=chunk,
            sampling_rate=hps.data.sampling_rate,
            hop_length=hps.data.hop_length,
            pbar_position=pbar_position,
            f0_method=f0_method,
            force_rebuild=force_rebuild,
            legacy_final_proj=hps.data.__dict__.get("contentvec_final_proj", True),
        )
        for (pbar_position, chunk) in enumerate(filepath_chunks)
    )
