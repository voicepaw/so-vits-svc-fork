import warnings
from logging import getLogger
from typing import Any, Literal, Sequence

import torch
from torch import Tensor, nn

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
from so_vits_svc_fork.modules.encoders import PosteriorEncoder, TextEncoder
from so_vits_svc_fork.modules.flows import ResidualCouplingBlock

from ..hparams import HParams

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
        type_: Literal[
            "hifi-gan",
            "istft",
            "ms-istft",
            "mb-istft",
            "ddsp-sins",
            "ddsp-combsub",
            "ddsp-combsubfast",
        ] = "hifi-gan",
        gen_istft_n_fft: int = 16,
        gen_istft_hop_size: int = 4,
        subbands: int = 4,
        encoder_n_layers: int = 16,
        flow_n_layers: int = 4,
        n_flows: int = 4,
        flow_kernel_size: int = 3,
        block_size: int = 512,
        n_harmonics: int = 128,
        n_mag_allpass: int = 256,
        n_mag_harmonic: int = 512,
        n_mag_noise: int = 256,
        bigvgan_h: HParams | None = None,
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
        self.n_layers_encoder = encoder_n_layers
        self.n_layers_flow = flow_n_layers
        self.n_flows = n_flows
        self.flow_kernel_size = flow_kernel_size
        self.subbands = subbands
        self.type_ = type_
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
            gin_channels=256,
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
            self._return_mb = False
        elif "istft" in type_:
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
        elif type_ in ["ddsp-sins", "ddsp-combsub", "ddsp-combsubfast"]:
            from .decoders.pc_ddsp import CombSub, CombSubFast, Sins

            if type_ == "ddsp-sins":
                self.dec = Sins(
                    sampling_rate=sampling_rate,
                    block_size=block_size,
                    n_harmonics=n_harmonics,
                    n_mag_allpass=n_mag_allpass,
                    n_mag_noise=n_mag_noise,
                    n_unit=inter_channels,
                    n_spk=n_speakers,
                )
            elif type_ == "ddsp-combsub":
                self.dec = CombSub(
                    sampling_rate=sampling_rate,
                    block_size=block_size,
                    n_mag_allpass=n_mag_allpass,
                    n_mag_harmonic=n_mag_harmonic,
                    n_mag_noise=n_mag_noise,
                    n_unit=inter_channels,
                    n_spk=n_speakers,
                )
            elif type_ == "ddsp-combsubfast":
                self.dec = CombSubFast(
                    sampling_rate=sampling_rate,
                    block_size=block_size,
                    n_unit=inter_channels,
                    n_spk=n_speakers,
                )
        elif type_ == "bigvgan":
            from .decoders.bigvgan import BigVGAN

            self.dec = BigVGAN(bigvgan_h)
        else:
            raise ValueError(f"Unknown type: {type_}")

        self.enc_q = PosteriorEncoder(
            in_channels=spec_channels,
            out_channels=inter_channels,
            hidden_channels=hidden_channels,
            kernel_size=flow_kernel_size,
            dilation_rate=1,
            n_layers=encoder_n_layers,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            channels=inter_channels,
            hidden_channels=hidden_channels,
            kernel_size=flow_kernel_size,
            dilation_rate=1,
            n_layers=flow_n_layers,
            n_flows=n_flows,
            gin_channels=gin_channels,
        )
        self.f0_decoder = F0Decoder(
            out_channels=1,
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
            spk_channels=gin_channels,
        )
        self.emb_uv = nn.Embedding(2, hidden_channels)

    def forward(
        self,
        c: Tensor,
        f0: Tensor,
        uv: Tensor,
        spec: Tensor,
        spk: Tensor | None = None,
        c_lengths: Tensor | None = None,
        spec_lengths: Tensor | None = None,
        volume: Tensor | None = None,
    ):
        # speaker embedding
        g = self.emb_g(spk).transpose(1, 2)

        # ssl prenet
        x_mask = torch.unsqueeze(commons.sequence_mask(c_lengths, c.size(2)), 1).to(
            c.dtype
        )
        x = self.pre(c) * x_mask + self.emb_uv(uv.long()).transpose(1, 2)

        # f0 decoder
        lf0 = 2595.0 * torch.log10(1.0 + f0.unsqueeze(1) / 700.0) / 500
        norm_lf0 = so_vits_svc_fork.f0.normalize_f0(lf0, x_mask, uv)
        pred_lf0 = self.f0_decoder(x, norm_lf0, x_mask, spk_emb=g)

        # posterior encoder
        _, m_p, logs_p, _ = self.enc_p(x, x_mask, f0=f0_to_coarse(f0))

        # spectrogram encoder
        z, m_q, logs_q, spec_mask = self.enc_q(spec, spec_lengths, g=g)

        # flow
        z_p = self.flow(z, spec_mask, g=g)

        # slice z, pitch with segment_size to decrease memory usage
        z_slice, pitch_slice, ids_slice = commons.rand_slice_segments_with_pitch(
            z, f0, spec_lengths, self.segment_size
        )

        # decoder
        o_mb = None
        if "istft" in self.type_:
            o, o_mb = self.dec(z_slice, g=g)
        elif "ddsp" in self.type_:
            o, _, (s_h, s_n) = self.dec(
                z_slice.transpose(1, 2),
                pitch_slice.unsqueeze(-1),
                volume.transpose(0, 1),
                spk.long(),
            )
        elif "bigvgan" in self.type_:
            o = self.dec(z_slice)
        else:
            o = self.dec(z_slice, g=g, f0=pitch_slice)
        return (
            o,
            o_mb,
            ids_slice,
            spec_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            pred_lf0,
            norm_lf0,
            lf0,
        )

    def infer(
        self,
        c: Tensor,
        f0: Tensor,
        uv: Tensor,
        spk: Tensor,
        noise_scale: float = 0.35,
        predict_f0: bool = False,
        volume: Tensor | None = None,
    ) -> Tensor:
        c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)

        # speaker embedding
        spk = self.emb_g(spk).transpose(1, 2)

        # ssl prenet
        x_mask = torch.unsqueeze(commons.sequence_mask(c_lengths, c.size(2)), 1).to(
            c.dtype
        )
        x = self.pre(c) * x_mask + self.emb_uv(uv.long()).transpose(1, 2)

        # f0 decoder
        if predict_f0:
            lf0 = 2595.0 * torch.log10(1.0 + f0.unsqueeze(1) / 700.0) / 500
            norm_lf0 = so_vits_svc_fork.f0.normalize_f0(
                lf0, x_mask, uv, random_scale=False
            )
            pred_lf0 = self.f0_decoder(x, norm_lf0, x_mask, spk_emb=spk)
            f0 = (700 * (torch.pow(10, pred_lf0 * 500 / 2595) - 1)).squeeze(1)

        # posterior encoder
        z_p, _, _, c_mask = self.enc_p(
            x, x_mask, f0=f0_to_coarse(f0), noise_scale=noise_scale
        )

        # flow (reverse)
        z = self.flow(z_p, c_mask, g=spk, reverse=True)

        # decoder
        if "istft" in self.type_:
            o, _ = self.dec(z * c_mask, g=spk)
        elif "ddsp" in self.type_:
            assert volume is not None
            o, _, _ = self.dec(
                (z * c_mask).transpose(1, 2),
                f0.unsqueeze(-1),
                volume.transpose(0, 1),
                spk.long(),
            )
        elif "bigvgan" in self.type_:
            o = self.dec(z)
        else:
            o = self.dec(z * c_mask, g=spk, f0=f0)
        return o
