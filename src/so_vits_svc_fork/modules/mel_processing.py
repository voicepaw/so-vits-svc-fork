"""from logging import getLogger

import torch
import torch.utils.data
import torchaudio

LOG = getLogger(__name__)


from ..hparams import HParams


def spectrogram_torch(audio: torch.Tensor, hps: HParams) -> torch.Tensor:
    return torchaudio.transforms.Spectrogram(
        n_fft=hps.data.filter_length,
        win_length=hps.data.win_length,
        hop_length=hps.data.hop_length,
        power=1.0,
        window_fn=torch.hann_window,
        normalized=False,
    ).to(audio.device)(audio)


def spec_to_mel_torch(spec: torch.Tensor, hps: HParams) -> torch.Tensor:
    return torchaudio.transforms.MelScale(
        n_mels=hps.data.n_mel_channels,
        sample_rate=hps.data.sampling_rate,
        f_min=hps.data.mel_fmin,
        f_max=hps.data.mel_fmax,
    ).to(spec.device)(spec)


def mel_spectrogram_torch(audio: torch.Tensor, hps: HParams) -> torch.Tensor:
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=hps.data.sampling_rate,
        n_fft=hps.data.filter_length,
        n_mels=hps.data.n_mel_channels,
        win_length=hps.data.win_length,
        hop_length=hps.data.hop_length,
        f_min=hps.data.mel_fmin,
        f_max=hps.data.mel_fmax,
        power=1.0,
        window_fn=torch.hann_window,
        normalized=False,
    ).to(audio.device)(audio)"""

from logging import getLogger

import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn

LOG = getLogger(__name__)

MAX_WAV_VALUE = 32768.0


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def spectrogram_torch(y, hps, center=False):
    if torch.min(y) < -1.0:
        LOG.info("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        LOG.info("max value is ", torch.max(y))
    n_fft = hps.data.filter_length
    hop_size = hps.data.hop_length
    win_size = hps.data.win_length
    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


def spec_to_mel_torch(spec, hps):
    sampling_rate = hps.data.sampling_rate
    n_fft = hps.data.filter_length
    num_mels = hps.data.n_mel_channels
    fmin = hps.data.mel_fmin
    fmax = hps.data.mel_fmax
    global mel_basis
    dtype_device = str(spec.dtype) + "_" + str(spec.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(
            dtype=spec.dtype, device=spec.device
        )
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)
    return spec


def mel_spectrogram_torch(y, hps, center=False):
    sampling_rate = hps.data.sampling_rate
    n_fft = hps.data.filter_length
    num_mels = hps.data.n_mel_channels
    fmin = hps.data.mel_fmin
    fmax = hps.data.mel_fmax
    hop_size = hps.data.hop_length
    win_size = hps.data.win_length
    if torch.min(y) < -1.0:
        LOG.info(f"min value is {torch.min(y)}")
    if torch.max(y) > 1.0:
        LOG.info(f"max value is {torch.max(y)}")

    global mel_basis, hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(
            dtype=y.dtype, device=y.device
        )
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)

    return spec
