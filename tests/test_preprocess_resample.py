import tempfile
from pathlib import Path

import numpy as np
import soundfile

from so_vits_svc_fork.preprocessing.preprocess_resample import preprocess_resample

SR = 44100
TONES = {"f0": 440.0, "f1": 550.0, "f2": 660.0}


def _write_tone(path: Path, freq: float) -> None:
    t = np.arange(SR) / SR
    soundfile.write(path, (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32), SR)


def _dominant_frequency(path: Path) -> float:
    audio, sr = soundfile.read(path)
    return float(np.argmax(np.abs(np.fft.rfft(audio))) * sr / len(audio))


def test_preprocess_resample_writes_each_audio_file_to_its_own_name() -> None:
    """A skipped entry must not shift the pairing of input and output paths.

    ``Path.rglob("*.*")`` also matches directories, and ``preprocess_resample``
    skips any entry that is not directly under the input directory.  Such an
    entry is absent from ``out_paths`` but was still present in ``in_paths``, so
    the positional ``zip`` of the two lists wrote every following file to the
    previous file's name and dropped the last file entirely.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "dataset_raw"
        speaker_dir = input_dir / "alice.v1"
        speaker_dir.mkdir(parents=True)
        for stem, freq in TONES.items():
            _write_tone(speaker_dir / f"{stem}.wav", freq)

        output_dir = Path(tmpdir) / "dataset" / "44k"
        preprocess_resample(input_dir, output_dir, SR, n_jobs=1)

        for stem, freq in TONES.items():
            output_path = output_dir / "alice.v1" / f"{stem}.wav"
            assert output_path.is_file(), f"{output_path} was not written"
            assert abs(_dominant_frequency(output_path) - freq) < 5, f"{output_path} holds the wrong audio"
