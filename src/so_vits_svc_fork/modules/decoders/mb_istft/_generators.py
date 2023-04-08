import math

import torch
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

from ....modules import modules
from ....modules.commons import get_padding, init_weights
from ._pqmf import PQMF
from ._stft import TorchSTFT


class iSTFT_Generator(torch.nn.Module):
    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gen_istft_n_fft,
        gen_istft_hop_size,
        gin_channels=0,
    ):
        super().__init__()
        # self.h = h
        self.gen_istft_n_fft = gen_istft_n_fft
        self.gen_istft_hop_size = gen_istft_hop_size

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = weight_norm(
            Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        )
        resblock = modules.ResBlock1 if resblock == "1" else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))

        self.post_n_fft = self.gen_istft_n_fft
        self.conv_post = weight_norm(Conv1d(ch, self.post_n_fft + 2, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.reflection_pad = torch.nn.ReflectionPad1d((1, 0))
        self.stft = TorchSTFT(
            filter_length=self.gen_istft_n_fft,
            hop_length=self.gen_istft_hop_size,
            win_length=self.gen_istft_n_fft,
        )

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.reflection_pad(x)
        x = self.conv_post(x)
        spec = torch.exp(x[:, : self.post_n_fft // 2 + 1, :])
        phase = math.pi * torch.sin(x[:, self.post_n_fft // 2 + 1 :, :])
        out = self.stft.inverse(spec, phase).to(x.device)
        return out, None

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class Multiband_iSTFT_Generator(torch.nn.Module):
    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gen_istft_n_fft,
        gen_istft_hop_size,
        subbands,
        gin_channels=0,
    ):
        super().__init__()
        # self.h = h
        self.subbands = subbands
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = weight_norm(
            Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        )
        resblock = modules.ResBlock1 if resblock == "1" else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))

        self.post_n_fft = gen_istft_n_fft
        self.ups.apply(init_weights)
        self.reflection_pad = torch.nn.ReflectionPad1d((1, 0))
        self.reshape_pixelshuffle = []

        self.subband_conv_post = weight_norm(
            Conv1d(ch, self.subbands * (self.post_n_fft + 2), 7, 1, padding=3)
        )

        self.subband_conv_post.apply(init_weights)

        self.gen_istft_n_fft = gen_istft_n_fft
        self.gen_istft_hop_size = gen_istft_hop_size

    def forward(self, x, g=None):
        stft = TorchSTFT(
            filter_length=self.gen_istft_n_fft,
            hop_length=self.gen_istft_hop_size,
            win_length=self.gen_istft_n_fft,
        ).to(x.device)
        pqmf = PQMF(x.device, subbands=self.subbands).to(x.device, dtype=x.dtype)

        x = self.conv_pre(x)  # [B, ch, length]

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.reflection_pad(x)
        x = self.subband_conv_post(x)
        x = torch.reshape(
            x, (x.shape[0], self.subbands, x.shape[1] // self.subbands, x.shape[-1])
        )

        spec = torch.exp(x[:, :, : self.post_n_fft // 2 + 1, :])
        phase = math.pi * torch.sin(x[:, :, self.post_n_fft // 2 + 1 :, :])

        y_mb_hat = stft.inverse(
            torch.reshape(
                spec,
                (
                    spec.shape[0] * self.subbands,
                    self.gen_istft_n_fft // 2 + 1,
                    spec.shape[-1],
                ),
            ),
            torch.reshape(
                phase,
                (
                    phase.shape[0] * self.subbands,
                    self.gen_istft_n_fft // 2 + 1,
                    phase.shape[-1],
                ),
            ),
        )
        y_mb_hat = torch.reshape(
            y_mb_hat, (x.shape[0], self.subbands, 1, y_mb_hat.shape[-1])
        )
        y_mb_hat = y_mb_hat.squeeze(-2)

        y_g_hat = pqmf.synthesis(y_mb_hat)

        return y_g_hat, y_mb_hat

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class Multistream_iSTFT_Generator(torch.nn.Module):
    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gen_istft_n_fft,
        gen_istft_hop_size,
        subbands,
        gin_channels=0,
    ):
        super().__init__()
        # self.h = h
        self.subbands = subbands
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = weight_norm(
            Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        )
        resblock = modules.ResBlock1 if resblock == "1" else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))

        self.post_n_fft = gen_istft_n_fft
        self.ups.apply(init_weights)
        self.reflection_pad = torch.nn.ReflectionPad1d((1, 0))
        self.reshape_pixelshuffle = []

        self.subband_conv_post = weight_norm(
            Conv1d(ch, self.subbands * (self.post_n_fft + 2), 7, 1, padding=3)
        )

        self.subband_conv_post.apply(init_weights)

        self.gen_istft_n_fft = gen_istft_n_fft
        self.gen_istft_hop_size = gen_istft_hop_size

        updown_filter = torch.zeros(
            (self.subbands, self.subbands, self.subbands)
        ).float()
        for k in range(self.subbands):
            updown_filter[k, k, 0] = 1.0
        self.register_buffer("updown_filter", updown_filter)
        self.multistream_conv_post = weight_norm(
            Conv1d(
                self.subbands, 1, kernel_size=63, bias=False, padding=get_padding(63, 1)
            )
        )
        self.multistream_conv_post.apply(init_weights)

    def forward(self, x, g=None):
        stft = TorchSTFT(
            filter_length=self.gen_istft_n_fft,
            hop_length=self.gen_istft_hop_size,
            win_length=self.gen_istft_n_fft,
        ).to(x.device)
        # pqmf = PQMF(x.device)

        x = self.conv_pre(x)  # [B, ch, length]

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.reflection_pad(x)
        x = self.subband_conv_post(x)
        x = torch.reshape(
            x, (x.shape[0], self.subbands, x.shape[1] // self.subbands, x.shape[-1])
        )

        spec = torch.exp(x[:, :, : self.post_n_fft // 2 + 1, :])
        phase = math.pi * torch.sin(x[:, :, self.post_n_fft // 2 + 1 :, :])

        y_mb_hat = stft.inverse(
            torch.reshape(
                spec,
                (
                    spec.shape[0] * self.subbands,
                    self.gen_istft_n_fft // 2 + 1,
                    spec.shape[-1],
                ),
            ),
            torch.reshape(
                phase,
                (
                    phase.shape[0] * self.subbands,
                    self.gen_istft_n_fft // 2 + 1,
                    phase.shape[-1],
                ),
            ),
        )
        y_mb_hat = torch.reshape(
            y_mb_hat, (x.shape[0], self.subbands, 1, y_mb_hat.shape[-1])
        )
        y_mb_hat = y_mb_hat.squeeze(-2)

        y_mb_hat = F.conv_transpose1d(
            y_mb_hat,
            self.updown_filter.to(x.device) * self.subbands,
            stride=self.subbands,
        )

        y_g_hat = self.multistream_conv_post(y_mb_hat)

        return y_g_hat, y_mb_hat

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
