import json
import os
import re
import warnings
import wave
from copy import deepcopy
from logging import getLogger
from pathlib import Path
from random import shuffle

from tqdm import tqdm

LOG = getLogger(__name__)


def _get_wav_duration(file_path):
    with wave.open(file_path, "rb") as wav_file:
        n_frames = wav_file.getnframes()
        framerate = wav_file.getframerate()
        duration = n_frames / float(framerate)
    return duration


def preprocess_config(
    input_dir: Path,
    train_list_path: Path,
    val_list_path: Path,
    test_list_path: Path,
):
    train = []
    val = []
    test = []
    spk_dict = {}
    spk_id = 0
    for speaker in tqdm(os.listdir(input_dir)):
        spk_dict[speaker] = spk_id
        spk_id += 1
        paths = [
            input_dir / speaker / i for i in (input_dir / speaker).glob("**/*.wav")
        ]
        new_paths = []
        for path in paths:
            pattern = re.compile(r"^[\.a-zA-Z0-9_\/]+$")
            if not pattern.match(path.name):
                warnings.warn(f"file name {path} contains non-alphanumeric characters.")
            if _get_wav_duration(path) < 0.3:
                warnings.warn(f"skip {path} because it is too short.")
                continue
            new_paths.append(path)
        paths = new_paths
        shuffle(paths)
        train += paths[2:-2]
        val += paths[:2]
        test += paths[-2:]

    LOG.info("Writing", train_list_path)
    with open(train_list_path, "w") as f:
        for fname in tqdm(train):
            wavpath = fname
            f.write(wavpath + "\n")

    LOG.info("Writing", val_list_path)
    with open(val_list_path, "w") as f:
        for fname in tqdm(val):
            wavpath = fname
            f.write(wavpath + "\n")

    LOG.info("Writing", test_list_path)
    with open(test_list_path, "w") as f:
        for fname in tqdm(test):
            wavpath = fname
            f.write(wavpath + "\n")

    config = deepcopy(
        json.loads(Path("configs_template/config_template.json").read_text())
    )
    config["spk"] = spk_dict
    LOG.info("Writing configs/config.json")
    with open("configs/config.json", "w") as f:
        json.dump(config, f, indent=2)
