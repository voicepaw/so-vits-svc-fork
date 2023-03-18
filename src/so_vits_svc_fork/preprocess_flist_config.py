from __future__ import annotations

import json
import os
import re
import wave
from copy import deepcopy
from logging import getLogger
from pathlib import Path
from random import shuffle

from tqdm import tqdm

LOG = getLogger(__name__)


def _get_wav_duration(filepath: Path):
    with open(filepath, "rb") as f:
        with wave.open(f) as wav_file:
            n_frames = wav_file.getnframes()
            framerate = wav_file.getframerate()
            duration = n_frames / float(framerate)
    return duration


def preprocess_config(
    input_dir: Path | str,
    train_list_path: Path | str,
    val_list_path: Path | str,
    test_list_path: Path | str,
    config_path: Path | str,
):
    input_dir = Path(input_dir)
    train_list_path = Path(train_list_path)
    val_list_path = Path(val_list_path)
    test_list_path = Path(test_list_path)
    config_path = Path(config_path)
    train = []
    val = []
    test = []
    spk_dict = {}
    spk_id = 0
    for speaker in os.listdir(input_dir):
        spk_dict[speaker] = spk_id
        spk_id += 1
        paths = []
        for path in tqdm(list((input_dir / speaker).glob("**/*.wav"))):
            pattern = re.compile(r"^[\.a-zA-Z0-9_\/]+$")
            if not pattern.match(path.name):
                LOG.warning(f"file name {path} contains non-alphanumeric characters.")
            if _get_wav_duration(path) < 0.3:
                LOG.warning(f"skip {path} because it is too short.")
                continue
            paths.append(path)
        shuffle(paths)
        if len(paths) <= 4:
            raise ValueError(
                f"too few files in {input_dir / speaker} (expected at least 5)."
            )
        train += paths[2:-2]
        val += paths[:2]
        test += paths[-2:]

    LOG.info(f"Writing {train_list_path}")
    train_list_path.parent.mkdir(parents=True, exist_ok=True)
    with train_list_path.open("w") as f:
        for fname in train:
            wavpath = fname.as_posix()
            f.write(wavpath + "\n")

    LOG.info(f"Writing {val_list_path}")
    val_list_path.parent.mkdir(parents=True, exist_ok=True)
    with val_list_path.open("w") as f:
        for fname in val:
            wavpath = fname.as_posix()
            f.write(wavpath + "\n")

    LOG.info(f"Writing {test_list_path}")
    test_list_path.parent.mkdir(parents=True, exist_ok=True)
    with test_list_path.open("w") as f:
        for fname in test:
            wavpath = fname.as_posix()
            f.write(wavpath + "\n")

    config = deepcopy(
        json.loads(
            (
                Path(__file__).parent / "configs_template" / "config_template.json"
            ).read_text()
        )
    )
    config["spk"] = spk_dict
    config["data"]["training_files"] = train_list_path.as_posix()
    config["data"]["validation_files"] = val_list_path.as_posix()
    LOG.info(f"Writing {config_path}")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w") as f:
        json.dump(config, f, indent=2)
