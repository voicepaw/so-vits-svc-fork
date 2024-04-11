from __future__ import annotations

from collections import defaultdict
from logging import getLogger
from pathlib import Path

import librosa
import soundfile as sf
import torch
from joblib import Parallel, delayed
from pyannote.audio import Pipeline
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

LOG = getLogger(__name__)


def _process_one(
    input_path: Path,
    output_dir: Path,
    sr: int,
    *,
    min_speakers: int = 1,
    max_speakers: int = 1,
    huggingface_token: str | None = None,
) -> None:
    try:
        audio, sr = librosa.load(input_path, sr=sr, mono=True)
    except Exception as e:
        LOG.warning(f"Failed to read {input_path}: {e}")
        return
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=huggingface_token
    )
    if pipeline is None:
        raise ValueError("Failed to load pipeline")
    pipeline = pipeline.to(torch.device("cuda"))
    LOG.info(f"Processing {input_path}. This may take a while...")
    diarization = pipeline(
        input_path, min_speakers=min_speakers, max_speakers=max_speakers
    )

    LOG.info(f"Found {len(diarization)} tracks, writing to {output_dir}")
    speaker_count = defaultdict(int)

    output_dir.mkdir(parents=True, exist_ok=True)
    for segment, track, speaker in tqdm(
        list(diarization.itertracks(yield_label=True)), desc=f"Writing {input_path}"
    ):
        if segment.end - segment.start < 1:
            continue
        speaker_count[speaker] += 1
        audio_cut = audio[int(segment.start * sr) : int(segment.end * sr)]
        sf.write(
            (output_dir / f"{speaker}_{speaker_count[speaker]:04d}.wav"),
            audio_cut,
            sr,
        )

    LOG.info(f"Speaker count: {speaker_count}")


def preprocess_speaker_diarization(
    input_dir: Path | str,
    output_dir: Path | str,
    sr: int,
    *,
    min_speakers: int = 1,
    max_speakers: int = 1,
    huggingface_token: str | None = None,
    n_jobs: int = -1,
) -> None:
    if huggingface_token is not None and not huggingface_token.startswith("hf_"):
        LOG.warning("Huggingface token probably should start with hf_")
    if not torch.cuda.is_available():
        LOG.warning("CUDA is not available. This will be extremely slow.")
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_paths = list(input_dir.rglob("*.*"))
    with tqdm_joblib(desc="Preprocessing speaker diarization", total=len(input_paths)):
        Parallel(n_jobs=n_jobs)(
            delayed(_process_one)(
                input_path,
                output_dir / input_path.relative_to(input_dir).parent / input_path.stem,
                sr,
                max_speakers=max_speakers,
                min_speakers=min_speakers,
                huggingface_token=huggingface_token,
            )
            for input_path in input_paths
        )
