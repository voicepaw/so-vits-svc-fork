from __future__ import annotations

import warnings
from logging import getLogger
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.accelerators import TPUAccelerator
from lightning.pytorch.loggers import TensorBoardLogger
from torch.cuda.amp import autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

import so_vits_svc_fork.f0
import so_vits_svc_fork.modules.commons as commons
import so_vits_svc_fork.utils

from . import utils
from .dataset import TextAudioCollate, TextAudioDataset
from .modules.descriminators import MultiPeriodDiscriminator
from .modules.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from .modules.mel_processing import mel_spectrogram_torch
from .modules.synthesizers import SynthesizerTrn

LOG = getLogger(__name__)
torch.backends.cudnn.benchmark = True


class VCDataModule(pl.LightningDataModule):
    def __init__(self, hparams: Any):
        super().__init__()
        self.__hparams = hparams
        self.collate_fn = TextAudioCollate()

        # these should be called in setup(), but we need to calculate check_val_every_n_epoch
        self.train_dataset = TextAudioDataset(self.__hparams, is_validation=False)
        self.val_dataset = TextAudioDataset(self.__hparams, is_validation=True)

    def train_dataloader(self):
        # since dataset just reads data from a file, set num_workers to 0
        return DataLoader(
            self.train_dataset,
            batch_size=self.__hparams.train.batch_size,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            collate_fn=self.collate_fn,
        )


def train(
    config_path: Path | str, model_path: Path | str, reset_optimizer: bool = False
):
    config_path = Path(config_path)
    model_path = Path(model_path)

    hparams = utils.get_backup_hparams(config_path, model_path)
    utils.ensure_pretrained_model(model_path, hparams.model.get("type_", "hifi-gan"))

    datamodule = VCDataModule(hparams)
    trainer = pl.Trainer(
        logger=TensorBoardLogger(model_path),
        # profiler="simple",
        val_check_interval=hparams.train.eval_interval,
        max_epochs=hparams.train.epochs,
        check_val_every_n_epoch=None,
        precision=16
        if hparams.train.fp16_run
        else "bf16"
        if hparams.train.get("bf16_run", False)
        else 32,
    )
    model = VitsLightning(reset_optimizer=reset_optimizer, **hparams)
    trainer.fit(model, datamodule=datamodule)


class VitsLightning(pl.LightningModule):
    def __init__(self, reset_optimizer: bool = False, **hparams: Any):
        super().__init__()
        self._temp_epoch = 0  # Add this line to initialize the _temp_epoch attribute
        self.save_hyperparameters("reset_optimizer")
        self.save_hyperparameters(*[k for k in hparams.keys()])
        torch.manual_seed(self.hparams.train.seed)
        self.net_g = SynthesizerTrn(
            self.hparams.data.filter_length // 2 + 1,
            self.hparams.train.segment_size // self.hparams.data.hop_length,
            **self.hparams.model,
        )
        self.net_d = MultiPeriodDiscriminator(self.hparams.model.use_spectral_norm)
        self.automatic_optimization = False
        self.optim_g = torch.optim.AdamW(
            self.net_g.parameters(),
            self.hparams.train.learning_rate,
            betas=self.hparams.train.betas,
            eps=self.hparams.train.eps,
        )
        self.optim_d = torch.optim.AdamW(
            self.net_d.parameters(),
            self.hparams.train.learning_rate,
            betas=self.hparams.train.betas,
            eps=self.hparams.train.eps,
        )
        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            self.optim_g, gamma=self.hparams.train.lr_decay
        )
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
            self.optim_d, gamma=self.hparams.train.lr_decay
        )
        self.optimizers_count = 2
        self.load(reset_optimizer)

    def on_train_start(self) -> None:
        self.set_current_epoch(self._temp_epoch)
        total_batch_idx = self._temp_epoch * len(self.trainer.train_dataloader)
        self.set_total_batch_idx(total_batch_idx)
        global_step = total_batch_idx * self.optimizers_count
        self.set_global_step(global_step)

        # check if using tpu
        if isinstance(self.trainer.accelerator, TPUAccelerator):
            # patch torch.stft to use cpu
            LOG.warning("Using TPU. Patching torch.stft to use cpu.")

            def stft(
                input: torch.Tensor,
                n_fft: int,
                hop_length: int | None = None,
                win_length: int | None = None,
                window: torch.Tensor | None = None,
                center: bool = True,
                pad_mode: str = "reflect",
                normalized: bool = False,
                onesided: bool | None = None,
                return_complex: bool | None = None,
            ) -> torch.Tensor:
                device = input.device
                input = input.cpu()
                if window is not None:
                    window = window.cpu()
                return torch.functional.stft(
                    input,
                    n_fft,
                    hop_length,
                    win_length,
                    window,
                    center,
                    pad_mode,
                    normalized,
                    onesided,
                    return_complex,
                ).to(device)

            torch.stft = stft

    def set_current_epoch(self, epoch: int):
        LOG.info(f"Setting current epoch to {epoch}")
        self.trainer.fit_loop.epoch_progress.current.completed = epoch
        assert self.current_epoch == epoch, f"{self.current_epoch} != {epoch}"

    def set_global_step(self, global_step: int):
        LOG.info(f"Setting global step to {global_step}")
        self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.total.completed = (
            global_step
        )
        self.trainer.fit_loop.epoch_loop.automatic_optimization.optim_progress.optimizer.step.total.completed = (
            global_step
        )
        assert self.global_step == global_step, f"{self.global_step} != {global_step}"

    def set_total_batch_idx(self, total_batch_idx: int):
        LOG.info(f"Setting total batch idx to {total_batch_idx}")
        self.trainer.fit_loop.epoch_loop.batch_progress.total.ready = (
            total_batch_idx + 1
        )
        self.trainer.fit_loop.epoch_loop.batch_progress.total.completed = (
            total_batch_idx
        )
        assert (
            self.total_batch_idx == total_batch_idx + 1
        ), f"{self.total_batch_idx} != {total_batch_idx + 1}"

    @property
    def total_batch_idx(self) -> int:
        return self.trainer.fit_loop.epoch_loop.total_batch_idx + 1

    def load(self, reset_optimizer: bool = False):
        latest_g_path = utils.latest_checkpoint_path(self.hparams.model_dir, "G_*.pth")
        latest_d_path = utils.latest_checkpoint_path(self.hparams.model_dir, "D_*.pth")
        if latest_g_path is not None and latest_d_path is not None:
            try:
                _, _, _, epoch = utils.load_checkpoint(
                    latest_g_path,
                    self.net_g,
                    self.optim_g,
                    reset_optimizer,
                )
                _, _, _, epoch = utils.load_checkpoint(
                    latest_d_path,
                    self.net_d,
                    self.optim_d,
                    reset_optimizer,
                )
                self._temp_epoch = epoch
                self.scheduler_g.last_epoch = epoch - 1
                self.scheduler_d.last_epoch = epoch - 1
            except Exception as e:
                raise RuntimeError("Failed to load checkpoint") from e
        else:
            LOG.warning("No checkpoint found. Start from scratch.")

    def configure_optimizers(self):
        return [self.optim_g, self.optim_d], [self.scheduler_g, self.scheduler_d]

    def log_image_dict(
        self, image_dict: dict[str, Any], dataformats: str = "HWC"
    ) -> None:
        if not isinstance(self.logger, TensorBoardLogger):
            warnings.warn("Image logging is only supported with TensorBoardLogger.")
            return
        writer: SummaryWriter = self.logger.experiment
        for k, v in image_dict.items():
            try:
                writer.add_image(k, v, self.total_batch_idx, dataformats=dataformats)
            except Exception as e:
                warnings.warn(f"Failed to log image {k}: {e}")

    def log_audio_dict(self, audio_dict: dict[str, Any]) -> None:
        if not isinstance(self.logger, TensorBoardLogger):
            warnings.warn("Audio logging is only supported with TensorBoardLogger.")
            return
        writer: SummaryWriter = self.logger.experiment
        for k, v in audio_dict.items():
            writer.add_audio(
                k,
                v,
                self.total_batch_idx,
                sample_rate=self.hparams.data.sampling_rate,
            )

    def log_dict_(self, log_dict: dict[str, Any], **kwargs) -> None:
        if not isinstance(self.logger, TensorBoardLogger):
            warnings.warn("Logging is only supported with TensorBoardLogger.")
            return
        writer: SummaryWriter = self.logger.experiment
        for k, v in log_dict.items():
            writer.add_scalar(k, v, self.total_batch_idx)
        kwargs["logger"] = False
        self.log_dict(log_dict, **kwargs)

    def log_(self, key: str, value: Any, **kwargs) -> None:
        self.log_dict_({key: value}, **kwargs)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        self.net_g.train()
        self.net_d.train()

        # get optims
        optim_g, optim_d = self.optimizers()

        # Generator
        # train
        self.toggle_optimizer(optim_g)
        c, f0, spec, mel, y, g, lengths, uv = batch
        (
            y_hat,
            y_hat_mb,
            ids_slice,
            z_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            pred_lf0,
            norm_lf0,
            lf0,
        ) = self.net_g(c, f0, uv, spec, g=g, c_lengths=lengths, spec_lengths=lengths)
        y_mel = commons.slice_segments(
            mel,
            ids_slice,
            self.hparams.train.segment_size // self.hparams.data.hop_length,
        )
        y_hat_mel = mel_spectrogram_torch(y_hat.squeeze(1), self.hparams)
        y = commons.slice_segments(
            y,
            ids_slice * self.hparams.data.hop_length,
            self.hparams.train.segment_size,
        )

        # generator loss
        LOG.debug("Calculating generator loss")
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.net_d(y, y_hat)

        with autocast(enabled=False):
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * self.hparams.train.c_mel
            loss_kl = (
                kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * self.hparams.train.c_kl
            )
            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, losses_gen = generator_loss(y_d_hat_g)
            loss_lf0 = F.mse_loss(pred_lf0, lf0)
            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl + loss_lf0

            # MB-iSTFT-VITS
            loss_subband = torch.tensor(0.0)
            if self.hparams.model.get("type_") == "mb-istft":
                from .modules.decoders.mb_istft import PQMF, subband_stft_loss

                y_mb = PQMF(y.device, self.hparams.model.subbands).analysis(y)
                loss_subband = subband_stft_loss(self.hparams, y_mb, y_hat_mb)
            loss_gen_all += loss_subband

        # log loss
        self.log_(
            "grad_norm_g", commons.clip_grad_value_(self.net_g.parameters(), None)
        )
        self.log_("lr", self.optim_g.param_groups[0]["lr"])
        self.log_dict_(
            {
                "loss/g/total": loss_gen_all,
                "loss/g/fm": loss_fm,
                "loss/g/mel": loss_mel,
                "loss/g/kl": loss_kl,
                "loss/g/lf0": loss_lf0,
            },
            prog_bar=True,
        )
        if self.hparams.model.get("type_") == "mb-istft":
            self.log_("loss/g/subband", loss_subband)
        if self.total_batch_idx % self.hparams.train.log_interval == 0:
            self.log_image_dict(
                {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(
                        y_mel[0].data.cpu().numpy()
                    ),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                        y_hat_mel[0].data.cpu().numpy()
                    ),
                    "all/mel": utils.plot_spectrogram_to_numpy(
                        mel[0].data.cpu().numpy()
                    ),
                    "all/lf0": so_vits_svc_fork.utils.plot_data_to_numpy(
                        lf0[0, 0, :].cpu().numpy(),
                        pred_lf0[0, 0, :].detach().cpu().numpy(),
                    ),
                    "all/norm_lf0": so_vits_svc_fork.utils.plot_data_to_numpy(
                        lf0[0, 0, :].cpu().numpy(),
                        norm_lf0[0, 0, :].detach().cpu().numpy(),
                    ),
                }
            )

        # optimizer
        optim_g.zero_grad()
        self.manual_backward(loss_gen_all)
        optim_g.step()
        self.untoggle_optimizer(optim_g)

        # Discriminator
        # train
        self.toggle_optimizer(optim_d)
        y_d_hat_r, y_d_hat_g, _, _ = self.net_d(y, y_hat.detach())

        # discriminator loss
        with autocast(enabled=False):
            loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                y_d_hat_r, y_d_hat_g
            )
            loss_disc_all = loss_disc

        # log loss
        self.log_("loss/d/total", loss_disc_all, prog_bar=True)
        self.log_(
            "grad_norm_d", commons.clip_grad_value_(self.net_d.parameters(), None)
        )

        # optimizer
        optim_d.zero_grad()
        self.manual_backward(loss_disc_all)
        optim_d.step()
        self.untoggle_optimizer(optim_d)

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            self.net_g.eval()
            c, f0, _, mel, y, g, _, uv = batch
            y_hat = self.net_g.infer(c, f0, uv, g=g)
            y_hat_mel = mel_spectrogram_torch(y_hat.squeeze(1).float(), self.hparams)
            self.log_audio_dict(
                {f"gen/audio_{batch_idx}": y_hat[0], f"gt/audio_{batch_idx}": y[0]}
            )
            self.log_image_dict(
                {
                    "gen/mel": utils.plot_spectrogram_to_numpy(
                        y_hat_mel[0].cpu().numpy()
                    ),
                    "gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy()),
                }
            )
            if self.current_epoch == 0 or batch_idx != 0:
                return
            utils.save_checkpoint(
                self.net_g,
                self.optim_g,
                self.hparams.train.learning_rate,
                self.current_epoch + 1,  # prioritize prevention of undervaluation
                Path(self.hparams.model_dir) / f"G_{self.total_batch_idx}.pth",
            )
            utils.save_checkpoint(
                self.net_d,
                self.optim_d,
                self.hparams.train.learning_rate,
                self.current_epoch + 1,
                Path(self.hparams.model_dir) / f"D_{self.total_batch_idx}.pth",
            )
            keep_ckpts = self.hparams.train.get("keep_ckpts", 0)
            if keep_ckpts > 0:
                utils.clean_checkpoints(
                    path_to_models=self.hparams.model_dir,
                    n_ckpts_to_keep=keep_ckpts,
                    sort_by_time=True,
                )
