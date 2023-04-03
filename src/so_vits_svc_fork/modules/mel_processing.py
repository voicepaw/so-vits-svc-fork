from logging import getLogger

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
        normalized=False,
        window_fn=torch.hann_window,
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
        normalized=False,
        window_fn=torch.hann_window,
    ).to(audio.device)(audio)
