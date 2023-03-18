from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib

# input_dir and output_dir exists.
# write code to convert input dir audio files to output dir audio files,
# without changing folder structure. Use joblib to parallelize.
# Converting audio files includes:
# - resampling to specified sampling rate
# - trim silence
# - adjust volume in a smart way
# - save as 16-bit wav file


def preprocess_resample(
    input_dir: Path | str, output_dir: Path | str, sampling_rate: int
) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    """Preprocess audio files in input_dir and save them to output_dir."""

    def preprocess_one(input_path: Path, output_path: Path) -> None:
        """Preprocess one audio file."""
        audio, sr = librosa.load(input_path)

        # Trim silence
        audio, _ = librosa.effects.trim(audio, top_db=20)

        # Adjust volume
        peak = np.abs(audio).max()
        if peak > 1.0:
            audio = 0.98 * audio / peak

        # Resample
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sampling_rate)
        audio /= max(audio.max(), -audio.min())
        soundfile.write(output_path, audio, samplerate=sampling_rate, subtype="PCM_16")

    in_and_out_paths = []
    for in_path in input_dir.rglob("*.wav"):
        out_path = output_dir / in_path.relative_to(input_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        in_and_out_paths.append((in_path, out_path))
    with tqdm_joblib(desc="Preprocessing", total=len(in_and_out_paths)):
        Parallel(n_jobs=-1)(delayed(preprocess_one)(*args) for args in in_and_out_paths)
