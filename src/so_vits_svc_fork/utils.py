from __future__ import annotations

import json
import re
import subprocess
from itertools import groupby
from logging import getLogger
from pathlib import Path
from typing import Any, Literal, Sequence

import matplotlib
import matplotlib.pylab as plt
import numpy as np
import requests
import torch
from cm_time import timer
from fairseq import checkpoint_utils
from fairseq.models.hubert.hubert import HubertModel
from numpy import ndarray
from scipy.io.wavfile import read
from tqdm import tqdm

from so_vits_svc_fork.hparams import HParams

LOG = getLogger(__name__)
HUBERT_SAMPLING_RATE = 16000


def download_file(
    url: str,
    filepath: Path | str,
    chunk_size: int = 64 * 1024,
    tqdm_cls: type = tqdm,
    skip_if_exists: bool = False,
    overwrite: bool = False,
    **tqdm_kwargs: Any,
):
    if skip_if_exists is True and overwrite is True:
        raise ValueError("skip_if_exists and overwrite cannot be both True")
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    temppath = filepath.parent / f"{filepath.name}.download"
    if filepath.exists():
        if skip_if_exists:
            return
        elif not overwrite:
            filepath.unlink()
        else:
            raise FileExistsError(f"{filepath} already exists")
    temppath.unlink(missing_ok=True)
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    kwargs = (
        dict(
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Downloading {filepath.name}",
        )
        | tqdm_kwargs
    )
    with temppath.open("wb") as f, tqdm_cls(**kwargs) as pbar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = f.write(data)
            pbar.update(size)
    temppath.rename(filepath)


PRETRAINED_MODEL_URLS = {
    "hifi-gan": [
        [
            "https://huggingface.co/therealvul/so-vits-svc-4.0-init/resolve/main/D_0.pth",
            "https://huggingface.co/therealvul/so-vits-svc-4.0-init/resolve/main/G_0.pth",
        ],
        [
            "https://huggingface.co/Himawari00/so-vits-svc4.0-pretrain-models/resolve/main/D_0.pth",
            "https://huggingface.co/Himawari00/so-vits-svc4.0-pretrain-models/resolve/main/G_0.pth",
        ],
    ],
    "contentvec": [
        [
            "https://huggingface.co/therealvul/so-vits-svc-4.0-init/resolve/main/checkpoint_best_legacy_500.pt"
        ],
        [
            "https://huggingface.co/Himawari00/so-vits-svc4.0-pretrain-models/resolve/main/checkpoint_best_legacy_500.pt"
        ],
        [
            "http://obs.cstcloud.cn/share/obs/sankagenkeshi/checkpoint_best_legacy_500.pt"
        ],
    ],
}
from joblib import Parallel, delayed


def ensure_pretrained_model(
    folder_path: Path | str, type_: str, **tqdm_kwargs: Any
) -> tuple[Path, ...] | None:
    folder_path = Path(folder_path)
    models_candidates = PRETRAINED_MODEL_URLS.get(type_, None)
    if models_candidates is None:
        LOG.warning(f"Unknown pretrained model type: {type_}")
        return
    for model_urls in models_candidates:
        paths = [folder_path / model_url.split("/")[-1] for model_url in model_urls]
        try:
            Parallel(n_jobs=len(paths))(
                [
                    delayed(download_file)(
                        url, path, position=i, skip_if_exists=True, **tqdm_kwargs
                    )
                    for i, (url, path) in enumerate(zip(model_urls, paths))
                ]
            )
            return tuple(paths)
        except Exception as e:
            LOG.exception(e)
    return


def get_hubert_model(device: torch.device) -> HubertModel:
    (path,) = ensure_pretrained_model(Path("."), "contentvec")

    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [path.as_posix()],
        suffix="",
    )
    model = models[0]
    return model.eval().to(device)


import warnings

import torchaudio


def get_content(
    cmodel: HubertModel,
    audio: torch.Tensor | ndarray[Any, Any],
    device: torch.device | str,
    sr: int,
    legacy_final_proj: bool = False,
) -> torch.Tensor:
    audio = torch.as_tensor(audio)
    if sr != HUBERT_SAMPLING_RATE:
        audio = torchaudio.transforms.Resample(sr, HUBERT_SAMPLING_RATE)(audio).to(
            device
        )
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    with torch.no_grad(), timer() as t:
        params = {"output_layer": 9} if legacy_final_proj else {}
        c: torch.Tensor = cmodel.extract_features(audio, **params)[0]
        if legacy_final_proj:
            warnings.warn("legacy_final_proj is deprecated")
            assert hasattr(cmodel, "final_proj")
            assert isinstance(cmodel.final_proj, torch.nn.Module)
            c = cmodel.final_proj(c)
        c = c.transpose(1, 2)
        # print(c.shape)
    wav_len = audio.shape[-1] / HUBERT_SAMPLING_RATE
    LOG.info(
        f"HuBERT inference time  : {t.elapsed:.3f}s, RTF: {t.elapsed / wav_len:.3f}"
    )
    return c


def load_checkpoint(
    checkpoint_path: Path | str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    skip_optimizer: bool = False,
) -> tuple[torch.nn.Module, torch.optim.Optimizer | None, float, int]:
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"File {checkpoint_path} not found")
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]
    if (
        optimizer is not None
        and not skip_optimizer
        and checkpoint_dict["optimizer"] is not None
    ):
        try:
            optimizer.load_state_dict(checkpoint_dict["optimizer"])
        except Exception as e:
            LOG.exception(e)
            LOG.warning("Failed to load optimizer state")
    saved_state_dict = checkpoint_dict["model"]
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
            assert saved_state_dict[k].shape == v.shape, (
                saved_state_dict[k].shape,
                v.shape,
            )
        except Exception as e:
            LOG.exception(e)
            LOG.error("%s is not in the checkpoint" % k)
            new_state_dict[k] = v
    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    LOG.info(f"Loaded checkpoint '{checkpoint_path}' (iteration {iteration})")
    return model, optimizer, learning_rate, iteration


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
    iteration: int,
    checkpoint_path: Path | str,
) -> None:
    LOG.info(
        "Saving model and optimizer state at iteration {} to {}".format(
            iteration, checkpoint_path
        )
    )
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(
        {
            "model": state_dict,
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path,
    )


def clean_checkpoints(
    path_to_models: Path | str, n_ckpts_to_keep: int = 2, sort_by_time: bool = True
) -> None:
    """Freeing up space by deleting saved ckpts

    Arguments:
    path_to_models    --  Path to the model directory
    n_ckpts_to_keep   --  Number of ckpts to keep, excluding G_0.pth and D_0.pth
    sort_by_time      --  True -> chronologically delete ckpts
                          False -> lexicographically delete ckpts
    """
    LOG.info("Cleaning old checkpoints...")
    path_to_models = Path(path_to_models)

    # Define sort key functions
    name_key = lambda p: int(re.match(r"[GD]_(\d+)", p.stem).group(1))
    time_key = lambda p: p.stat().st_mtime
    path_key = lambda p: (p.stem[0], time_key(p) if sort_by_time else name_key(p))

    models = list(
        filter(
            lambda p: (
                p.is_file()
                and re.match(r"[GD]_\d+", p.stem)
                and not p.stem.endswith("_0")
            ),
            path_to_models.glob("*.pth"),
        )
    )

    models_sorted = sorted(models, key=path_key)

    models_sorted_grouped = groupby(models_sorted, lambda p: p.stem[0])

    for group_name, group_items in models_sorted_grouped:
        to_delete_list = list(group_items)[:-n_ckpts_to_keep]

        for to_delete in to_delete_list:
            LOG.info(f"Removing {to_delete}")
            to_delete.unlink()


from torch.utils.tensorboard.writer import SummaryWriter


def summarize(
    writer: SummaryWriter,
    global_step: int,
    scalars: dict[str, float] = {},
    histograms: dict[str, ndarray] = {},
    images: dict[str, ndarray] = {},
    audios: dict[str, ndarray] = {},
    audio_sampling_rate: int | None = None,
) -> None:
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        if audio_sampling_rate is None:
            raise ValueError("audio_sampling_rate must be provided")
        writer.add_audio(k, v, global_step, audio_sampling_rate)


def latest_checkpoint_path(dir_path: Path | str, regex: str = "G_*.pth") -> Path | None:
    dir_path = Path(dir_path)
    name_key = lambda p: int(re.match(r"._(\d+)\.pth", p.name).group(1))
    paths = list(sorted(dir_path.glob(regex), key=name_key))
    if len(paths) == 0:
        return None
    return paths[-1]


def plot_spectrogram_to_numpy(spectrogram: ndarray) -> ndarray:
    matplotlib.use("Agg")
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def load_wav_to_torch(full_path: Path | str) -> tuple[torch.Tensor, int]:
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename: Path | str, split="|"):
    with open(filename, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def get_backup_hparams(
    config_path: Path, model_path: Path, init: bool = True
) -> HParams:
    model_path.mkdir(parents=True, exist_ok=True)
    config_save_path = model_path / "config.json"
    if init:
        with config_path.open() as f:
            data = f.read()
        with config_save_path.open("w") as f:
            f.write(data)
    else:
        with config_save_path.open() as f:
            data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    hparams.model_dir = model_path.as_posix()
    return hparams


def get_hparams(config_path: Path | str) -> HParams:
    config = json.loads(Path(config_path).read_text())
    hparams = HParams(**config)
    return hparams


def repeat_expand_2d(content: torch.Tensor, target_len: int) -> torch.Tensor:
    # content : [h, t]
    src_len = content.shape[-1]
    target = torch.zeros(
        [content.shape[0], target_len], dtype=content.dtype, device=content.device
    )
    temp = torch.arange(src_len + 1) * target_len / src_len
    current_pos = 0
    for i in range(target_len):
        if i < temp[current_pos + 1]:
            target[:, i] = content[:, current_pos]
        else:
            current_pos += 1
            target[:, i] = content[:, current_pos]
    return target


def plot_data_to_numpy(x: ndarray, y: ndarray) -> ndarray:
    matplotlib.use("Agg")
    fig, ax = plt.subplots(figsize=(10, 2))
    plt.plot(x)
    plt.plot(y)
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def get_gpu_memory(type_: Literal["total", "free", "used"]) -> Sequence[int]:
    command = f"nvidia-smi --query-gpu=memory.{type_} --format=csv"
    memory_free_info = (
        subprocess.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
    )
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


def get_total_gpu_memory(type_: Literal["total", "free", "used"]) -> int:
    return sum(get_gpu_memory(type_))
