from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def slice_segments(x: Tensor, starts: Tensor, length: int) -> Tensor:
    if length is None:
        return x
    length = min(length, x.size(-1))
    x_slice = torch.zeros((x.size()[:-1] + (length,)), dtype=x.dtype, device=x.device)
    ends = starts + length
    for i, (start, end) in enumerate(zip(starts, ends)):
        # LOG.debug(i, start, end, x.size(), x[i, ..., start:end].size(), x_slice.size())
        # x_slice[i, ...] = x[i, ..., start:end] need to pad
        # x_slice[i, ..., :end - start] = x[i, ..., start:end] this does not work
        x_slice[i, ...] = F.pad(x[i, ..., start:end], (0, max(0, length - x.size(-1))))
    return x_slice


def rand_slice_segments_with_pitch(
    x: Tensor, f0: Tensor, x_lengths: Tensor | int | None, segment_size: int | None
):
    if segment_size is None:
        return x, f0, torch.arange(x.size(0), device=x.device)
    if x_lengths is None:
        x_lengths = x.size(-1) * torch.ones(
            x.size(0), dtype=torch.long, device=x.device
        )
    # slice_starts = (torch.rand(z.size(0), device=z.device) * (z_lengths - segment_size)).long()
    slice_starts = (
        torch.rand(x.size(0), device=x.device)
        * torch.max(
            x_lengths - segment_size, torch.zeros_like(x_lengths, device=x.device)
        )
    ).long()
    z_slice = slice_segments(x, slice_starts, segment_size)
    f0_slice = slice_segments(f0, slice_starts, segment_size)
    return z_slice, f0_slice, slice_starts


def slice_2d_segments(x: Tensor, starts: Tensor, length: int) -> Tensor:
    batch_size, num_features, seq_len = x.shape
    ends = starts + length
    idxs = (
        torch.arange(seq_len, device=x.device)
        .unsqueeze(0)
        .unsqueeze(1)
        .repeat(batch_size, num_features, 1)
    )
    mask = (idxs >= starts.unsqueeze(-1).unsqueeze(-1)) & (
        idxs < ends.unsqueeze(-1).unsqueeze(-1)
    )
    return x[mask].reshape(batch_size, num_features, length)


def slice_1d_segments(x: Tensor, starts: Tensor, length: int) -> Tensor:
    batch_size, seq_len = x.shape
    ends = starts + length
    idxs = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
    mask = (idxs >= starts.unsqueeze(-1)) & (idxs < ends.unsqueeze(-1))
    return x[mask].reshape(batch_size, length)


def _slice_segments_v3(x: Tensor, starts: Tensor, length: int) -> Tensor:
    shape = x.shape[:-1] + (length,)
    ends = starts + length
    idxs = torch.arange(x.shape[-1], device=x.device).unsqueeze(0).unsqueeze(0)
    unsqueeze_dims = len(shape) - len(
        x.shape
    )  # calculate number of dimensions to unsqueeze
    starts = starts.reshape(starts.shape + (1,) * unsqueeze_dims)
    ends = ends.reshape(ends.shape + (1,) * unsqueeze_dims)
    mask = (idxs >= starts) & (idxs < ends)
    return x[mask].reshape(shape)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def subsequent_mask(length):
    mask = torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)
    return mask


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def clip_grad_value_(parameters, clip_value, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm
