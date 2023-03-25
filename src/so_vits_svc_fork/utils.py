from __future__ import annotations

import json
import re
from itertools import groupby
from logging import getLogger
from pathlib import Path
from typing import Any, Literal

import numpy as np
import requests
import torch
import torchcrepe
from cm_time import timer
from numpy import dtype, float32, ndarray
from scipy.io.wavfile import read
from torch import FloatTensor, Tensor
from tqdm import tqdm

LOG = getLogger(__name__)
MATPLOTLIB_FLAG = False
f0_bin = 256
f0_max = 1100.0
f0_min = 50.0
f0_mel_min = 1127 * np.log(1 + f0_min / 700)
f0_mel_max = 1127 * np.log(1 + f0_max / 700)
HUBERT_SAMPLING_RATE = 16000


# def normalize_f0(f0, random_scale=True):
#     f0_norm = f0.clone()  # create a copy of the input Tensor
#     batch_size, _, frame_length = f0_norm.shape
#     for i in range(batch_size):
#         means = torch.mean(f0_norm[i, 0, :])
#         if random_scale:
#             factor = random.uniform(0.8, 1.2)
#         else:
#             factor = 1
#         f0_norm[i, 0, :] = (f0_norm[i, 0, :] - means) * factor
#     return f0_norm
# def normalize_f0(f0, random_scale=True):
#     means = torch.mean(f0[:, 0, :], dim=1, keepdim=True)
#     if random_scale:
#         factor = torch.Tensor(f0.shape[0],1).uniform_(0.8, 1.2).to(f0.device)
#     else:
#         factor = torch.ones(f0.shape[0], 1, 1).to(f0.device)
#     f0_norm = (f0 - means.unsqueeze(-1)) * factor.unsqueeze(-1)
#     return f0_norm
def normalize_f0(
    f0: FloatTensor, x_mask: FloatTensor, uv: FloatTensor, random_scale=True
) -> FloatTensor:
    # calculate means based on x_mask
    uv_sum = torch.sum(uv, dim=1, keepdim=True)
    uv_sum[uv_sum == 0] = 9999
    means = torch.sum(f0[:, 0, :] * uv, dim=1, keepdim=True) / uv_sum

    if random_scale:
        factor = torch.Tensor(f0.shape[0], 1).uniform_(0.8, 1.2).to(f0.device)
    else:
        factor = torch.ones(f0.shape[0], 1).to(f0.device)
    # normalize f0 based on means and factor
    f0_norm = (f0 - means.unsqueeze(-1)) * factor.unsqueeze(-1)
    if torch.isnan(f0_norm).any():
        exit(0)
    return f0_norm * x_mask


def plot_data_to_numpy(x: ndarray, y: ndarray) -> ndarray:
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    plt.plot(x)
    plt.plot(y)
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def interpolate_f0(
    f0: ndarray[Any, dtype[float32]]
) -> tuple[ndarray[Any, dtype[float32]], ndarray[Any, dtype[float32]]]:
    data = np.reshape(f0, (f0.size, 1))

    vuv_vector = np.zeros((data.size, 1), dtype=np.float32)
    vuv_vector[data > 0.0] = 1.0
    vuv_vector[data <= 0.0] = 0.0

    ip_data = data

    frame_number = data.size
    last_value = 0.0
    for i in range(frame_number):
        if data[i] <= 0.0:
            j = i + 1
            for j in range(i + 1, frame_number):
                if data[j] > 0.0:
                    break
            if j < frame_number - 1:
                if last_value > 0.0:
                    step = (data[j] - data[i - 1]) / float(j - i)
                    for k in range(i, j):
                        ip_data[k] = data[i - 1] + step * (k - i + 1)
                else:
                    for k in range(i, j):
                        ip_data[k] = data[j]
            else:
                for k in range(i, frame_number):
                    ip_data[k] = last_value
        else:
            ip_data[i] = data[i]
            last_value = data[i]

    return ip_data[:, 0], vuv_vector[:, 0]


def compute_f0_parselmouth(
    wav_numpy: ndarray[Any, dtype[float32]],
    p_len: None | int = None,
    sampling_rate: int = 44100,
    hop_length: int = 512,
):
    import parselmouth

    x = wav_numpy
    if p_len is None:
        p_len = x.shape[0] // hop_length
    else:
        assert abs(p_len - x.shape[0] // hop_length) < 4, "pad length error"
    time_step = hop_length / sampling_rate * 1000
    f0_min = 50
    f0_max = 1100
    f0 = (
        parselmouth.Sound(x, sampling_rate)
        .to_pitch_ac(
            time_step=time_step / 1000,
            voicing_threshold=0.6,
            pitch_floor=f0_min,
            pitch_ceiling=f0_max,
        )
        .selected_array["frequency"]
    )

    pad_size = (p_len - len(f0) + 1) // 2
    if pad_size > 0 or p_len - len(f0) - pad_size > 0:
        f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant")
    return f0


def _resize_f0(
    x: ndarray[Any, dtype[float32]], target_len: int
) -> ndarray[Any, dtype[float32]]:
    source = np.array(x)
    source[source < 0.001] = np.nan
    target = np.interp(
        np.arange(0, len(source) * target_len, len(source)) / target_len,
        np.arange(0, len(source)),
        source,
    )
    res = np.nan_to_num(target)
    return res


def compute_f0_pyworld(
    wav_numpy: ndarray[Any, dtype[float32]],
    p_len: None | int = None,
    sampling_rate: int = 44100,
    hop_length: int = 512,
    type_: Literal["dio", "harvest"] = "dio",
):
    import pyworld

    if p_len is None:
        p_len = wav_numpy.shape[0] // hop_length
    if type_ == "dio":
        f0, t = pyworld.dio(
            wav_numpy.astype(np.double),
            fs=sampling_rate,
            f0_ceil=f0_max,
            f0_floor=f0_min,
            frame_period=1000 * hop_length / sampling_rate,
        )
    elif type_ == "harvest":
        f0, t = pyworld.harvest(
            wav_numpy.astype(np.double),
            fs=sampling_rate,
            f0_ceil=f0_max,
            f0_floor=f0_min,
            frame_period=1000 * hop_length / sampling_rate,
        )
    f0 = pyworld.stonemask(wav_numpy.astype(np.double), f0, t, sampling_rate)
    for index, pitch in enumerate(f0):
        f0[index] = round(pitch, 1)
    return _resize_f0(f0, p_len)


def compute_f0_crepe(
    wav_numpy: ndarray[Any, dtype[float32]],
    p_len: None | int = None,
    sampling_rate: int = 44100,
    hop_length: int = 512,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    model: Literal["full", "tiny"] = "full",
):
    audio = torch.from_numpy(wav_numpy).to(device, copy=True)
    audio = torch.unsqueeze(audio, dim=0)

    if audio.ndim == 2 and audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True).detach()
    # (T) -> (1, T)
    audio = audio.detach()

    pitch: Tensor = torchcrepe.predict(
        audio,
        sampling_rate,
        hop_length,
        f0_min,
        f0_max,
        model,
        batch_size=hop_length * 2,
        device=device,
        pad=True,
    )

    f0 = pitch.squeeze(0).cpu().numpy()
    p_len = p_len or wav_numpy.shape[0] // hop_length
    f0 = _resize_f0(f0, p_len)
    return f0


def compute_f0(
    wav_numpy: ndarray[Any, dtype[float32]],
    p_len: None | int = None,
    sampling_rate: int = 44100,
    hop_length: int = 512,
    method: Literal["crepe", "crepe-tiny", "parselmouth", "dio", "harvest"] = "dio",
    **kwargs,
):
    with timer() as t:
        wav_numpy = wav_numpy.astype(np.float32)
        wav_numpy /= np.quantile(np.abs(wav_numpy), 0.999)
        if method in ["dio", "harvest"]:
            f0 = compute_f0_pyworld(wav_numpy, p_len, sampling_rate, hop_length, method)
        elif method == "crepe":
            f0 = compute_f0_crepe(wav_numpy, p_len, sampling_rate, hop_length, **kwargs)
        elif method == "crepe-tiny":
            f0 = compute_f0_crepe(
                wav_numpy, p_len, sampling_rate, hop_length, model="tiny", **kwargs
            )
        elif method == "parselmouth":
            f0 = compute_f0_parselmouth(wav_numpy, p_len, sampling_rate, hop_length)
        else:
            raise ValueError(
                "type must be dio, crepe, crepe-tiny, harvest or parselmouth"
            )
    rtf = t.elapsed / (len(wav_numpy) / sampling_rate)
    LOG.info(f"F0 inference time:       {t.elapsed:.3f}s, RTF: {rtf:.3f}")
    return f0


def f0_to_coarse(f0: torch.Tensor | float):
    is_torch = isinstance(f0, torch.Tensor)
    f0_mel = 1127 * (1 + f0 / 700).log() if is_torch else 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (
        f0_mel_max - f0_mel_min
    ) + 1

    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    f0_coarse = (f0_mel + 0.5).long() if is_torch else np.rint(f0_mel).astype(np.int)
    assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
        f0_coarse.max(),
        f0_coarse.min(),
    )
    return f0_coarse


def download_file(
    url: str,
    filepath: Path | str,
    chunk_size: int = 4 * 1024,
    tqdm_cls: type = tqdm,
    **kwargs,
):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    temppath = filepath.parent / f"{filepath.name}.download"
    if filepath.exists():
        raise FileExistsError(f"{filepath} already exists")
    temppath.unlink(missing_ok=True)
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with temppath.open("wb") as f, tqdm_cls(
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
        **kwargs,
    ) as pbar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = f.write(data)
            pbar.update(size)
    temppath.rename(filepath)


def ensure_pretrained_model(folder_path: Path, **kwargs) -> None:
    model_urls = [
        # "https://huggingface.co/innnky/sovits_pretrained/resolve/main/sovits4/G_0.pth",
        "https://huggingface.co/therealvul/so-vits-svc-4.0-init/resolve/main/D_0.pth",
        # "https://huggingface.co/innnky/sovits_pretrained/resolve/main/sovits4/D_0.pth",
        "https://huggingface.co/therealvul/so-vits-svc-4.0-init/resolve/main/G_0.pth",
    ]
    for model_url in model_urls:
        model_path = folder_path / model_url.split("/")[-1]
        if not model_path.exists():
            download_file(
                model_url, model_path, desc=f"Downloading {model_path.name}", **kwargs
            )


def ensure_hubert_model(**kwargs) -> Path:
    vec_path = Path("checkpoint_best_legacy_500.pt")
    vec_path.parent.mkdir(parents=True, exist_ok=True)
    if not vec_path.exists():
        # url = "http://obs.cstcloud.cn/share/obs/sankagenkeshi/checkpoint_best_legacy_500.pt"
        # url = "https://huggingface.co/innnky/contentvec/resolve/main/checkpoint_best_legacy_500.pt"
        url = "https://huggingface.co/therealvul/so-vits-svc-4.0-init/resolve/main/checkpoint_best_legacy_500.pt"
        download_file(url, vec_path, desc="Downloading Hubert model", **kwargs)
    return vec_path


def get_hubert_model():
    vec_path = ensure_hubert_model()
    from fairseq import checkpoint_utils

    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [vec_path.as_posix()],
        suffix="",
    )
    model = models[0]
    model.eval()
    return model


def get_hubert_content(hmodel, wav_16k_tensor):
    with timer() as t:
        feats = wav_16k_tensor
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        inputs = {
            "source": feats.to(wav_16k_tensor.device),
            "padding_mask": padding_mask.to(wav_16k_tensor.device),
            "output_layer": 9,  # layer 9
        }
        with torch.no_grad():
            logits = hmodel.extract_features(**inputs)
            feats = hmodel.final_proj(logits[0])
        res = feats.transpose(1, 2)
    wav_len = wav_16k_tensor.shape[-1] / 16000
    LOG.info(
        f"HuBERT inference time  : {t.elapsed:.3f}s, RTF: {t.elapsed / wav_len:.3f}"
    )
    return res


def get_content(cmodel: Any, y: ndarray) -> ndarray:
    with torch.no_grad():
        c = cmodel.extract_features(y.squeeze(1))[0]
    c = c.transpose(1, 2)
    return c


def load_checkpoint(
    checkpoint_path: Any,
    model: Any,
    optimizer: Any = None,
    skip_optimizer: bool = False,
):
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
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    saved_state_dict = checkpoint_dict["model"]
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            # assert "dec" in k or "disc" in k
            # print("load", k)
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
    model, optimizer, learning_rate, iteration, checkpoint_path
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
):
    """Freeing up space by deleting saved ckpts

    Arguments:
    path_to_models    --  Path to the model directory
    n_ckpts_to_keep   --  Number of ckpts to keep, excluding G_0.pth and D_0.pth
    sort_by_time      --  True -> chronologically delete ckpts
                          False -> lexicographically delete ckpts
    """
    LOG.warning("Cleaning old checkpoints...")
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
            LOG.warning(f"Removing {to_delete}")
            to_delete.unlink()


def summarize(
    writer,
    global_step,
    scalars={},
    histograms={},
    images={},
    audios={},
    audio_sampling_rate=22050,
):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)


def latest_checkpoint_path(dir_path: Path | str, regex: str = "G_*.pth"):
    dir_path = Path(dir_path)
    name_key = lambda p: int(re.match(r"._(\d+)\.pth", p.name).group(1))
    return list(sorted(dir_path.glob(regex), key=name_key))[-1]


def plot_spectrogram_to_numpy(spectrogram):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
    import matplotlib.pylab as plt
    import numpy as np

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


def load_wav_to_torch(full_path: Path | str):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename: Path | str, split="|"):
    with open(filename, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def get_hparams(config_path: Path, model_path: Path, init: bool = True) -> HParams:
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


def get_hparams_from_file(config_path: Path | str) -> HParams:
    config = json.loads(Path(config_path).read_text())
    hparams = HParams(**config)
    return hparams


def repeat_expand_2d(content: ndarray, target_len: int) -> ndarray:
    # content : [h, t]

    src_len = content.shape[-1]
    target = torch.zeros([content.shape[0], target_len], dtype=torch.float).to(
        content.device
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


class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()
