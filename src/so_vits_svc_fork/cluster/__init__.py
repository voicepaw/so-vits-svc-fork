from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from sklearn.cluster import KMeans


def get_cluster_model(ckpt_path: Path | str):
    with Path(ckpt_path).open("rb") as f:
        checkpoint = torch.load(
            f, map_location="cpu"
        )  # Danger of arbitrary code execution
    kmeans_dict = {}
    for spk, ckpt in checkpoint.items():
        km = KMeans(ckpt["n_features_in_"])
        km.__dict__["n_features_in_"] = ckpt["n_features_in_"]
        km.__dict__["_n_threads"] = ckpt["_n_threads"]
        km.__dict__["cluster_centers_"] = ckpt["cluster_centers_"]
        kmeans_dict[spk] = km
    return kmeans_dict


def check_speaker(model: Any, speaker: Any):
    if speaker not in model:
        raise ValueError(f"Speaker {speaker} not in {list(model.keys())}")


def get_cluster_result(model: Any, x: Any, speaker: Any):
    """
    x: np.array [t, 256]
    return cluster class result
    """
    check_speaker(model, speaker)
    return model[speaker].predict(x)


def get_cluster_center_result(model: Any, x: Any, speaker: Any):
    """x: np.array [t, 256]"""
    check_speaker(model, speaker)
    predict = model[speaker].predict(x)
    return model[speaker].cluster_centers_[predict]


def get_center(model: Any, x: Any, speaker: Any):
    check_speaker(model, speaker)
    return model[speaker].cluster_centers_[x]
