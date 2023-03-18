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


def preprocess_hubert_f0(input_dir: Path | str, config_path: Path | str):
    input_dir = Path(input_dir)
    config_path = Path(config_path)
    utils.get_hubert_model()
    hps = utils.get_hparams_from_file(config_path)
    sampling_rate = hps.data.sampling_rate
    hop_length = hps.data.hop_length

    def _process_one(filepath: Path, hmodel, device: Literal["cuda", "cpu"] = "cuda"):
        wav, sr = librosa.load(filepath, sr=sampling_rate)
        soft_path = filepath.parent / (filepath.name + ".soft.pt")
        if not soft_path.exists():
            wav16k = librosa.resample(
                wav, orig_sr=sampling_rate, target_sr=HUBERT_SAMPLING_RATE
            )
            wav16k = torch.from_numpy(wav16k).to(device)
            c = utils.get_hubert_content(hmodel, wav_16k_tensor=wav16k)
            torch.save(c.cpu(), soft_path)
        f0_path = filepath.parent / (filepath.name + ".f0.npy")
        if not f0_path.exists():
            f0 = utils.compute_f0_dio(
                wav, sampling_rate=sampling_rate, hop_length=hop_length
            )
            np.save(f0_path, f0)

    def _process_batch(filepaths: Iterable[Path]):
        LOG.info("Loading hubert model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        hmodel = utils.get_hubert_model().to(device)
        LOG.info("Hubert model loaded.")
        for filepath in tqdm(filepaths):
            _process_one(filepath, hmodel, device)

    filepaths = list(input_dir.glob("**/*.wav"))
    n_jobs = min(cpu_count(), len(filepaths) // 32 + 1, 8)
    shuffle(filepaths)
    filepath_chunks = np.array_split(filepaths, n_jobs)
    Parallel(n_jobs=n_jobs)(delayed(_process_batch)(chunk) for chunk in filepath_chunks)
