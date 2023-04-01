from ._stft_loss import MultiResolutionSTFTLoss


def subband_stft_loss(h, y_mb, y_hat_mb):
    sub_stft_loss = MultiResolutionSTFTLoss(
        h.train.fft_sizes, h.train.hop_sizes, h.train.win_lengths
    )
    y_mb = y_mb.view(-1, y_mb.size(2))
    y_hat_mb = y_hat_mb.view(-1, y_hat_mb.size(2))
    sub_sc_loss, sub_mag_loss = sub_stft_loss(y_hat_mb[:, : y_mb.size(-1)], y_mb)
    return sub_sc_loss + sub_mag_loss
