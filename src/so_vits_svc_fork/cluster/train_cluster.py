from __future__ import annotations

import math
from logging import getLogger
from pathlib import Path
from typing import Any

import numpy as np
import torch
from cm_time import timer
from joblib import Parallel, delayed
from sklearn.cluster import KMeans, MiniBatchKMeans
from tqdm_joblib import tqdm_joblib

LOG = getLogger(__name__)


def train_cluster(
    input_dir: Path | str,
    n_clusters: int,
    use_minibatch: bool = True,
    batch_size: int = 4096,
    partial_fit: bool = False,
    verbose: bool = False,
) -> dict:
    input_dir = Path(input_dir)
    if not partial_fit:
        LOG.info(f"Loading features from {input_dir}")
        features = []
        for path in input_dir.rglob("*.data.pt"):
            with path.open("rb") as f:
                features.append(
                    torch.load(f, weights_only=True)["content"].squeeze(0).numpy().T
                )
        if not features:
            raise ValueError(f"No features found in {input_dir}")
        features = np.concatenate(features, axis=0).astype(np.float32)
        if features.shape[0] < n_clusters:
            raise ValueError(
                "Too few HuBERT features to cluster. Consider using a smaller number of clusters."
            )
        LOG.info(
            f"shape: {features.shape}, size: {features.nbytes/1024**2:.2f} MB, dtype: {features.dtype}"
        )
        with timer() as t:
            if use_minibatch:
                kmeans = MiniBatchKMeans(
                    n_clusters=n_clusters,
                    verbose=verbose,
                    batch_size=batch_size,
                    max_iter=80,
                    n_init="auto",
                ).fit(features)
            else:
                kmeans = KMeans(
                    n_clusters=n_clusters, verbose=verbose, n_init="auto"
                ).fit(features)
        LOG.info(f"Clustering took {t.elapsed:.2f} seconds")

        x = {
            "n_features_in_": kmeans.n_features_in_,
            "_n_threads": kmeans._n_threads,
            "cluster_centers_": kmeans.cluster_centers_,
        }
        return x
    else:
        # minibatch partial fit
        paths = list(input_dir.rglob("*.data.pt"))
        if len(paths) == 0:
            raise ValueError(f"No features found in {input_dir}")
        LOG.info(f"Found {len(paths)} features in {input_dir}")
        n_batches = math.ceil(len(paths) / batch_size)
        LOG.info(f"Splitting into {n_batches} batches")
        with timer() as t:
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                verbose=verbose,
                batch_size=batch_size,
                max_iter=80,
                n_init="auto",
            )
            for i in range(0, len(paths), batch_size):
                LOG.info(
                    f"Processing batch {i//batch_size+1}/{n_batches} for speaker {input_dir.stem}"
                )
                features = []
                for path in paths[i : i + batch_size]:
                    with path.open("rb") as f:
                        features.append(
                            torch.load(f, weights_only=True)["content"]
                            .squeeze(0)
                            .numpy()
                            .T
                        )
                features = np.concatenate(features, axis=0).astype(np.float32)
                kmeans.partial_fit(features)
        LOG.info(f"Clustering took {t.elapsed:.2f} seconds")

        x = {
            "n_features_in_": kmeans.n_features_in_,
            "_n_threads": kmeans._n_threads,
            "cluster_centers_": kmeans.cluster_centers_,
        }
        return x


def main(
    input_dir: Path | str,
    output_path: Path | str,
    n_clusters: int = 10000,
    use_minibatch: bool = True,
    batch_size: int = 4096,
    partial_fit: bool = False,
    verbose: bool = False,
) -> None:
    input_dir = Path(input_dir)
    output_path = Path(output_path)

    if not (use_minibatch or not partial_fit):
        raise ValueError("partial_fit requires use_minibatch")

    def train_cluster_(input_path: Path, **kwargs: Any) -> tuple[str, dict]:
        return input_path.stem, train_cluster(input_path, **kwargs)

    with tqdm_joblib(desc="Training clusters", total=len(list(input_dir.iterdir()))):
        parallel_result = Parallel(n_jobs=-1)(
            delayed(train_cluster_)(
                speaker_name,
                n_clusters=n_clusters,
                use_minibatch=use_minibatch,
                batch_size=batch_size,
                partial_fit=partial_fit,
                verbose=verbose,
            )
            for speaker_name in input_dir.iterdir()
        )
    assert parallel_result is not None
    checkpoint = dict(parallel_result)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with output_path.open("wb") as f:
        torch.save(checkpoint, f)
