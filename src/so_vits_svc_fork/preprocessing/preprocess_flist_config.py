from __future__ import annotations

import json
import os
from copy import deepcopy
from logging import getLogger
from pathlib import Path

import numpy as np
from librosa import get_duration
from tqdm import tqdm

LOG = getLogger(__name__)
CONFIG_TEMPLATE_DIR = Path(__file__).parent / "config_templates"


def preprocess_config(
    input_dir: Path | str,
    train_list_path: Path | str,
    val_list_path: Path | str,
    test_list_path: Path | str,
    config_path: Path | str,
    config_name: str,
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
    random = np.random.RandomState(1234)
    for speaker in os.listdir(input_dir):
        spk_dict[speaker] = spk_id
        spk_id += 1
        paths = []
        for path in tqdm(list((input_dir / speaker).rglob("*.wav"))):
            if get_duration(filename=path) < 0.3:
                LOG.warning(f"skip {path} because it is too short.")
                continue
            paths.append(path)
        random.shuffle(paths)
        if len(paths) <= 4:
            raise ValueError(
                f"too few files in {input_dir / speaker} (expected at least 5)."
            )
        train += paths[2:-2]
        val += paths[:2]
        test += paths[-2:]

    LOG.info(f"Writing {train_list_path}")
    train_list_path.parent.mkdir(parents=True, exist_ok=True)
    train_list_path.write_text(
        "\n".join([x.as_posix() for x in train]), encoding="utf-8"
    )

    LOG.info(f"Writing {val_list_path}")
    val_list_path.parent.mkdir(parents=True, exist_ok=True)
    val_list_path.write_text("\n".join([x.as_posix() for x in val]), encoding="utf-8")

    LOG.info(f"Writing {test_list_path}")
    test_list_path.parent.mkdir(parents=True, exist_ok=True)
    test_list_path.write_text("\n".join([x.as_posix() for x in test]), encoding="utf-8")

    config = deepcopy(
        json.loads(
            (
                CONFIG_TEMPLATE_DIR
                / (
                    config_name
                    if config_name.endswith(".json")
                    else config_name + ".json"
                )
            ).read_text(encoding="utf-8")
        )
    )
    config["spk"] = spk_dict
    config["data"]["training_files"] = train_list_path.as_posix()
    config["data"]["validation_files"] = val_list_path.as_posix()
    LOG.info(f"Writing {config_path}")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
