import random
import warnings
from logging import getLogger
from typing import Any, Literal, Sequence

import torch
from torch import nn

import so_vits_svc_fork.f0
from so_vits_svc_fork.f0 import f0_to_coarse
from so_vits_svc_fork.modules import commons as commons
from so_vits_svc_fork.modules.decoders.f0 import F0Decoder
from so_vits_svc_fork.modules.decoders.hifigan import NSFHifiGANGenerator
from so_vits_svc_fork.modules.decoders.mb_istft import (
    Multiband_iSTFT_Generator,
    Multistream_iSTFT_Generator,
    iSTFT_Generator,
)
from so_vits_svc_fork.modules.encoders import Encoder, TextEncoder
from so_vits_svc_fork.modules.flows import ResidualCouplingBlock

LOG = getLogger(__name__)


class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: int,
        resblock: str,
        resblock_kernel_sizes: Sequence[int],
        resblock_dilation_sizes: Sequence[Sequence[int]],
        upsample_rates: Sequence[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: Sequence[int],
        gin_channels: int,
        ssl_dim: int,
        n_speakers: int,
        sampling_rate: int = 44100,
        type_: Literal["hifi-gan", "istft", "ms-istft", "mb-istft"] = "hifi-gan",
        gen_istft_n_fft: int = 16,
        gen_istft_hop_size: int = 4,
        subbands: int = 4,
        ipu: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.ssl_dim = ssl_dim
        self.n_speakers = n_speakers
        self.sampling_rate = sampling_rate
        self.type_ = type_
        self.gen_istft_n_fft = gen_istft_n_fft
        self.gen_istft_hop_size = gen_istft_hop_size
        self.subbands = subbands
        if kwargs:
            warnings.warn(f"Unused arguments: {kwargs}")

        self.emb_g = nn.Embedding(n_speakers, gin_channels)

        if ssl_dim is None:
            self.pre = nn.LazyConv1d(hidden_channels, kernel_size=5, padding=2)
        else:
            self.pre = nn.Conv1d(ssl_dim, hidden_channels, kernel_size=5, padding=2)

        self.enc_p = TextEncoder(
            inter_channels,
            hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
        )

        LOG.info(f"Decoder type: {type_}")
        if type_ == "hifi-gan":
            hps = {
                "sampling_rate": sampling_rate,
                "inter_channels": inter_channels,
                "resblock": resblock,
                "resblock_kernel_sizes": resblock_kernel_sizes,
                "resblock_dilation_sizes": resblock_dilation_sizes,
                "upsample_rates": upsample_rates,
                "upsample_initial_channel": upsample_initial_channel,
                "upsample_kernel_sizes": upsample_kernel_sizes,
                "gin_channels": gin_channels,
            }
            self.dec = NSFHifiGANGenerator(h=hps)
            self.mb = False
        else:
            hps = {
                "initial_channel": inter_channels,
                "resblock": resblock,
                "resblock_kernel_sizes": resblock_kernel_sizes,
                "resblock_dilation_sizes": resblock_dilation_sizes,
                "upsample_rates": upsample_rates,
                "upsample_initial_channel": upsample_initial_channel,
                "upsample_kernel_sizes": upsample_kernel_sizes,
                "gin_channels": gin_channels,
                "gen_istft_n_fft": gen_istft_n_fft,
                "gen_istft_hop_size": gen_istft_hop_size,
                "subbands": subbands,
            }

            # gen_istft_n_fft, gen_istft_hop_size, subbands
            if type_ == "istft":
                del hps["subbands"]
                self.dec = iSTFT_Generator(**hps)
            elif type_ == "ms-istft":
                self.dec = Multistream_iSTFT_Generator(**hps)
            elif type_ == "mb-istft":
                self.dec = Multiband_iSTFT_Generator(**hps)
            else:
                raise ValueError(f"Unknown type: {type_}")
            self.mb = True

        self.enc_q = Encoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels
        )
        self.f0_decoder = F0Decoder(
            1,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            spk_channels=gin_channels,
        )
        self.emb_uv = nn.Embedding(2, hidden_channels)
        self.ipu = ipu

    def forward(
        self,
        c: torch.Tensor,
        f0: torch.Tensor,
        uv: torch.Tensor,
        spec: torch.Tensor,
        g: torch.Tensor,
        c_lengths: torch.Tensor,
        spec_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        B: batch size
        c: content, (B, ssl_dim, T)
        f0: f0, (B, T)
        uv: uv, (B, T)
        spec: spectrogram, (B, F, T)
        g: speaker id, (B,)
        c_lengths: content length, (B,)
        spec_lengths: spectrogram length, (B,)
        """
        g = self.emb_g(g).transpose(1, 2)
        # ssl prenet
        x_mask = torch.unsqueeze(commons.sequence_mask(c_lengths, c.size(2)), 1).to(
            c.dtype
        )
        x = self.pre(c) * x_mask + self.emb_uv(uv.long()).transpose(1, 2)

        # f0 predict
        lf0 = 2595.0 * torch.log10(1.0 + f0.unsqueeze(1) / 700.0) / 500
        norm_lf0 = so_vits_svc_fork.f0.normalize_f0(lf0, x_mask, uv)
        pred_lf0 = self.f0_decoder(x, norm_lf0, x_mask, spk_emb=g)

        # encoder
        z_ptemp, m_p, logs_p, _ = self.enc_p(x, x_mask, f0=f0_to_coarse(f0))
        z, m_q, logs_q, spec_mask = self.enc_q(spec, spec_lengths, g=g)
        # z: [batch, dim, time]

        # flow
        z_p = self.flow(z, spec_mask, g=g)

        # randomly slice to time = self.segment_size
        if not self.ipu:
            slice_starts = (
                torch.rand(z.size(0)) * (spec_lengths - self.segment_size)
            ).long()
            z_slice = commons.slice_2d_segments(z, slice_starts, self.segment_size)
            f0_slice = commons.slice_1d_segments(f0, slice_starts, self.segment_size)
        else:
            slice_starts = random.randint(0, spec.size(2) - self.segment_size)
            z_slice = z[..., slice_starts : slice_starts + self.segment_size]
            f0_slice = f0[..., slice_starts : slice_starts + self.segment_size]

        # MB-iSTFT-VITS
        if self.mb:
            o, o_mb = self.dec(z_slice, g=g)
        # HiFi-GAN
        else:
            o = self.dec(z_slice, g=g, f0=f0_slice)
            o_mb = None
        return (
            o,
            o_mb,
            slice_starts,
            spec_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            pred_lf0,
            norm_lf0,
            lf0,
        )

    def infer(
        self,
        c: torch.Tensor,
        f0: torch.Tensor,
        uv: torch.Tensor,
        g: torch.Tensor,
        noice_scale: float = 0.35,
        predict_f0: bool = False,
    ) -> torch.Tensor:
        c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)
        g = self.emb_g(g).transpose(1, 2)
        x_mask = torch.unsqueeze(commons.sequence_mask(c_lengths, c.size(2)), 1).to(
            c.dtype
        )
        x = self.pre(c) * x_mask + self.emb_uv(uv.long()).transpose(1, 2)

        if predict_f0:
            lf0 = 2595.0 * torch.log10(1.0 + f0.unsqueeze(1) / 700.0) / 500
            norm_lf0 = so_vits_svc_fork.f0.normalize_f0(
                lf0, x_mask, uv, random_scale=False
            )
            pred_lf0 = self.f0_decoder(x, norm_lf0, x_mask, spk_emb=g)
            f0 = (700 * (torch.pow(10, pred_lf0 * 500 / 2595) - 1)).squeeze(1)

        z_p, m_p, logs_p, c_mask = self.enc_p(
            x, x_mask, f0=f0_to_coarse(f0), noice_scale=noice_scale
        )
        z = self.flow(z_p, c_mask, g=g, reverse=True)

        # MB-iSTFT-VITS
        if self.mb:
            o, o_mb = self.dec(z * c_mask, g=g)
        else:
            o = self.dec(z * c_mask, g=g, f0=f0)
        return o
