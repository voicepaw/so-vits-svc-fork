from __future__ import annotations

from logging import getLogger
from pathlib import Path
from typing import Any

import numpy as np
import torch
from cm_time import timer
from joblib import Parallel, delayed
from sklearn.cluster import KMeans, MiniBatchKMeans
from tqdm import tqdm

LOG = getLogger(__name__)


def train_cluster(
    input_dir: Path | str,
    n_clusters: int,
    use_minibatch: bool = True,
    verbose: bool = False,
) -> dict:
    input_dir = Path(input_dir)
    LOG.info(f"Loading features from {input_dir}")
    features = []
    nums = 0
    for path in tqdm(input_dir.glob("*.soft.pt")):
        features.append(torch.load(path).squeeze(0).numpy().T)
    features = np.concatenate(features, axis=0).astype(np.float32)
    LOG.info(
        f"Nums: {nums}, shape: {features.shape}, size: {features.nbytes/1024**2:.2f} MB, dtype: {features.dtype}"
    )
    with timer() as t:
        if use_minibatch:
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters, verbose=verbose, batch_size=4096, max_iter=80
            ).fit(features)
        else:
            kmeans = KMeans(n_clusters=n_clusters, verbose=verbose).fit(features)
    LOG.info(f"Clustering took {t.elapsed:.2f} seconds")

    x = {
        "n_features_in_": kmeans.n_features_in_,
        "_n_threads": kmeans._n_threads,
        "cluster_centers_": kmeans.cluster_centers_,
    }
    return x


def main(
    input_dir: Path | str, output_path: Path | str, n_clusters: int = 10000
) -> None:
    input_dir = Path(input_dir)
    output_path = Path(output_path)

    def train_cluster_(input_path: Path, **kwargs: Any) -> tuple[str, dict]:
        return input_path.stem, train_cluster(input_path, **kwargs)

    parallel_result = Parallel(n_jobs=-1)(
        delayed(train_cluster_)(input_path_, n_clusters=n_clusters)
        for input_path_ in input_dir.iterdir()
    )
    assert parallel_result is not None
    checkpoint = dict(parallel_result)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(checkpoint, output_path)
