from __future__ import annotations

from logging import getLogger
from pathlib import Path
from random import shuffle
from typing import Iterable, Literal

import librosa
import numpy as np
import torch
from joblib import Parallel, cpu_count, delayed
from tqdm import tqdm

from . import utils
from .utils import HUBERT_SAMPLING_RATE

LOG = getLogger(__name__)


def _process_one(
    filepath: Path,
    hubert_model,
    sampling_rate: int,
    hop_length: int,
    device: Literal["cuda", "cpu"] = "cuda",
    f0_method: Literal[
        "crepe", "crepe-tiny", "parselmouth", "dio", "harvest"
    ] = "crepe",
    force_rebuild: bool = False,
):
    wav, sr = librosa.load(filepath, sr=sampling_rate)

    # Compute HuBERT content
    soft_path = filepath.parent / (filepath.name + ".soft.pt")
    if not soft_path.exists() or force_rebuild:
        wav16k = librosa.resample(
            wav, orig_sr=sampling_rate, target_sr=HUBERT_SAMPLING_RATE
        )
        wav16k = torch.from_numpy(wav16k).to(device)
        c = utils.get_hubert_content(hubert_model, wav_16k_tensor=wav16k)
        torch.save(c.cpu(), soft_path)
    else:
        LOG.info(f"Skip {filepath} because {soft_path} exists.")

    # Compute f0
    f0_path = filepath.parent / (filepath.name + ".f0.npy")
    if not f0_path.exists() or force_rebuild:
        f0 = utils.compute_f0(
            wav, sampling_rate=sampling_rate, hop_length=hop_length, method=f0_method
        )
        np.save(f0_path, f0)
    else:
        LOG.info(f"Skip {filepath} because {f0_path} exists.")
    torch.cuda.empty_cache()


def _process_batch(
    filepaths: Iterable[Path],
    sampling_rate: int,
    hop_length: int,
    pbar_position: int,
    f0_method: Literal[
        "crepe", "crepe-tiny", "parselmouth", "dio", "harvest"
    ] = "crepe",
    force_rebuild: bool = False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hubert_model = utils.get_hubert_model().to(device)

    for filepath in tqdm(filepaths, position=pbar_position):
        _process_one(
            filepath,
            hubert_model,
            sampling_rate,
            hop_length,
            device,
            f0_method,
            force_rebuild,
        )


def preprocess_hubert_f0(
    input_dir: Path | str,
    config_path: Path | str,
    n_jobs: int = 4,
    f0_method: Literal[
        "crepe", "crepe-tiny", "parselmouth", "dio", "harvest"
    ] = "crepe",
    force_rebuild: bool = False,
):
    input_dir = Path(input_dir)
    config_path = Path(config_path)
    utils.ensure_hubert_model()
    hps = utils.get_hparams_from_file(config_path)
    sampling_rate = hps.data.sampling_rate
    hop_length = hps.data.hop_length

    filepaths = list(input_dir.rglob("*.wav"))
    n_jobs = min(cpu_count(), len(filepaths) // 32 + 1, n_jobs)
    shuffle(filepaths)
    filepath_chunks = np.array_split(filepaths, n_jobs)
    Parallel(n_jobs=n_jobs)(
        delayed(_process_batch)(
            chunk, sampling_rate, hop_length, pbar_position, f0_method, force_rebuild
        )
        for (pbar_position, chunk) in enumerate(filepath_chunks)
    )
