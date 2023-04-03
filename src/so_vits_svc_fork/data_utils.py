import random
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
import torch.utils.data

# import h5py


# Multi speaker version


class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    def __init__(self, audiopaths, hps):
        self.datapaths = [
            Path(x).parent / (Path(x).name + ".data.pt")
            for x in Path(hps.data.training_files).read_text().splitlines()
        ]
        random.seed(hps.train.seed)
        random.shuffle(self.datapaths)

    def __getitem__(self, index):
        return torch.load(self.datapaths[index], weights_only=True)

    def __len__(self):
        return len(self.datapaths)


def _pad_stack(array: Sequence[torch.Tensor]) -> torch.Tensor:
    max_idx = torch.argmax(torch.tensor([x_.shape[-1] for x_ in array]))
    max_x = array[max_idx]
    x_padded = [
        F.pad(x_, (0, max_x.shape[-1] - x_.shape[-1]), mode="constant", value=0)
        for x_ in array
    ]
    return torch.stack(x_padded)


class TextAudioCollate:
    """This code uses torch.stack() to create tensors from a list of tensors,
    torch.scatter_() to copy values from a source tensor to a destination tensor based on indices,
    and unsqueeze() to reshape tensors to match the dimensions of the destination tensor.
    """

    def __call__(self, batch):
        batch = [b for b in batch if b is not None]
        lengths = torch.tensor([b["mel_spec"].shape[1] for b in batch]).long()
        batch = list(sorted(batch, key=lambda x: x["mel_spec"].shape[1], reverse=True))
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
