from __future__ import annotations

from pathlib import Path
from random import Random
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from .hparams import HParams


class TextAudioDataset(Dataset):
    def __init__(self, hps: HParams, is_validation: bool = False):
        self.datapaths = [
            Path(x).parent / (Path(x).name + ".data.pt")
            for x in Path(
                hps.data.validation_files if is_validation else hps.data.training_files
            )
            .read_text("utf-8")
            .splitlines()
        ]
        self.hps = hps
        self.random = Random(hps.train.seed)
        self.random.shuffle(self.datapaths)
        self.max_spec_len = 800

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        with Path(self.datapaths[index]).open("rb") as f:
            data = torch.load(f, weights_only=True, map_location="cpu")

        # cut long data randomly
        spec_len = data["mel_spec"].shape[1]
        hop_len = self.hps.data.hop_length
        if spec_len > self.max_spec_len:
            start = self.random.randint(0, spec_len - self.max_spec_len)
            end = start + self.max_spec_len - 10
            for key in data.keys():
                if key == "audio":
                    data[key] = data[key][:, start * hop_len : end * hop_len]
                elif key == "spk":
                    continue
                else:
                    data[key] = data[key][..., start:end]
        torch.cuda.empty_cache()
        return data

    def __len__(self) -> int:
        return len(self.datapaths)


def _pad_stack(array: Sequence[torch.Tensor]) -> torch.Tensor:
    max_idx = torch.argmax(torch.tensor([x_.shape[-1] for x_ in array]))
    max_x = array[max_idx]
    x_padded = [
        F.pad(x_, (0, max_x.shape[-1] - x_.shape[-1]), mode="constant", value=0)
        for x_ in array
    ]
    return torch.stack(x_padded)


class TextAudioCollate(nn.Module):
    def forward(
        self, batch: Sequence[dict[str, torch.Tensor]]
    ) -> tuple[torch.Tensor, ...]:
        batch = [b for b in batch if b is not None]
        batch = list(sorted(batch, key=lambda x: x["mel_spec"].shape[1], reverse=True))
        lengths = torch.tensor([b["mel_spec"].shape[1] for b in batch]).long()
        results = {}
        for key in batch[0].keys():
            if key not in ["spk"]:
                results[key] = _pad_stack([b[key] for b in batch]).cpu()
            else:
                results[key] = torch.tensor([[b[key]] for b in batch]).cpu()

        return (
            results["content"],
            results["f0"],
            results["spec"],
            results["mel_spec"],
            results["audio"],
            results["spk"],
            lengths,
            results["uv"],
        )
