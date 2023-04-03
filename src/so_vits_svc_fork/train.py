from __future__ import annotations

import multiprocessing
import os
import time
from logging import getLogger
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

import so_vits_svc_fork.f0
import so_vits_svc_fork.modules.commons as commons
import so_vits_svc_fork.utils

from . import utils
from .data_utils import TextAudioCollate, TextAudioSpeakerLoader
from .hparams import HParams
from .modules.descriminators import MultiPeriodDiscriminator
from .modules.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from .modules.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from .modules.synthesizers import SynthesizerTrn

LOG = getLogger(__name__)
torch.backends.cudnn.benchmark = True
global_step = 0
start_time = time.time()


def train(
    config_path: Path | str, model_path: Path | str, reset_optimizer: bool = False
):
    """Assume Single Node Multi GPUs Training Only"""
    config_path = Path(config_path)
    model_path = Path(model_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    hps = utils.get_backup_hparams(config_path, model_path)
    utils.ensure_pretrained_model(model_path, hps.model.get("type_", "hifi-gan"))
    n_gpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = hps.train.port

    mp.spawn(
        _run,
        nprocs=n_gpus,
        args=(n_gpus, hps, reset_optimizer),
    )


def _run(rank: int, n_gpus: int, hps: HParams, reset_optimizer: bool = False):
    global global_step
    if rank == 0:
        LOG.info(hps)
        # utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=Path(hps.model_dir) / "eval")

    # for pytorch on win, backend use gloo
    dist.init_process_group(
        backend="gloo" if os.name == "nt" else "nccl",
        init_method="env://",
        world_size=n_gpus,
        rank=rank,
    )
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)
    collate_fn = TextAudioCollate()
    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps)
    num_workers = 5 if multiprocessing.cpu_count() > 4 else multiprocessing.cpu_count()
    train_loader = DataLoader(
        train_dataset,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        batch_size=hps.train.batch_size,
        collate_fn=collate_fn,
    )
    if rank == 0:
        eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps)
        eval_loader = DataLoader(
            eval_dataset,
            num_workers=1,
            shuffle=False,
            batch_size=1,
            pin_memory=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    ).cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    net_g = DDP(net_g, device_ids=[rank])
    net_d = DDP(net_d, device_ids=[rank])

    latest_g_path = utils.latest_checkpoint_path(hps.model_dir, "G_*.pth")
    latest_d_path = utils.latest_checkpoint_path(hps.model_dir, "D_*.pth")
    if latest_g_path is not None and latest_d_path is not None:
        try:
            _, _, _, epoch_str = utils.load_checkpoint(
                latest_g_path,
                net_g,
                optim_g,
                reset_optimizer,
            )
            _, _, _, epoch_str = utils.load_checkpoint(
                latest_d_path,
                net_d,
                optim_d,
                reset_optimizer,
            )
            epoch_str = max(epoch_str, 1)
            global_step = (epoch_str - 1) * len(train_loader)
        except Exception as e:
            raise RuntimeError("Failed to load checkpoint") from e
    else:
        LOG.warning("No checkpoint found. Start from scratch.")
        epoch_str = 1
        global_step = 0
    if reset_optimizer:
        epoch_str = 1
        global_step = 0

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )

    scaler = GradScaler(enabled=hps.train.fp16_run)

    LOG.info(
        "Start training..."
        "Note: You do not need to wait until the progress bar is full."
    )

    for epoch in trange(
        epoch_str, hps.train.epochs + 1, initial=epoch_str, total=hps.train.epochs
    ):
        if rank == 0:
            _train_and_evaluate(
                rank,
                epoch,
                hps,
                (net_g, net_d),
                (optim_g, optim_d),
                (scheduler_g, scheduler_d),
                scaler,
                (train_loader, eval_loader),
                (writer, writer_eval),
            )
        else:
            _train_and_evaluate(
                rank,
                epoch,
                hps,
                (net_g, net_d),
                (optim_g, optim_d),
                (scheduler_g, scheduler_d),
                scaler,
                (train_loader, None),
                None,
            )
        scheduler_g.step()
        scheduler_d.step()


def _train_and_evaluate(
    rank: int,
    epoch: int,
    hps: HParams,
    nets: tuple[nn.Module, nn.Module],
    optims: tuple[torch.optim.Optimizer, torch.optim.Optimizer],
    schedulers: tuple[
        torch.optim.lr_scheduler.ExponentialLR, torch.optim.lr_scheduler.ExponentialLR
    ],
    scaler: GradScaler,
    loaders: tuple[DataLoader, DataLoader | None],
    writers: None | tuple[SummaryWriter, SummaryWriter],
):
    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    # train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()
    for batch_idx, items in enumerate(train_loader):
        c, f0, spec, y, spk, lengths, uv = items
        g = spk.cuda(rank, non_blocking=True)
        spec, y = spec.cuda(rank, non_blocking=True), y.cuda(rank, non_blocking=True)
        c = c.cuda(rank, non_blocking=True)
        f0 = f0.cuda(rank, non_blocking=True)
        uv = uv.cuda(rank, non_blocking=True)
        lengths = lengths.cuda(rank, non_blocking=True)
        mel = spec_to_mel_torch(
            spec,
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.mel_fmin,
            hps.data.mel_fmax,
        )

        with autocast(enabled=hps.train.fp16_run):
            (
                y_hat,
                y_hat_mb,
                ids_slice,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
                pred_lf0,
                norm_lf0,
                lf0,
            ) = net_g(c, f0, uv, spec, g=g, c_lengths=lengths, spec_lengths=lengths)

            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y = commons.slice_segments(
                y, ids_slice * hps.data.hop_length, hps.train.segment_size
            )  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())

            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
                loss_disc_all = loss_disc

        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            with autocast(enabled=False):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_lf0 = F.mse_loss(pred_lf0, lf0)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl + loss_lf0

                # MB-iSTFT-VITS
                loss_subband = torch.tensor(0.0)
                if hps.model.get("type_") == "mb-istft":
                    from .modules.decoders.mb_istft import PQMF, subband_stft_loss

                    y_mb = PQMF(y.device, hps.model.subbands).analysis(y)
                    loss_subband = subband_stft_loss(hps, y_mb, y_hat_mb)
                loss_gen_all += loss_subband

        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                losses = {
                    "discriminator": loss_disc.item(),
                    "generator": loss_gen.item(),
                    "feature_matching": loss_fm.item(),
                    "melspectrogram": loss_mel.item(),
                    "kl_divergence": loss_kl.item(),
                }
                if hps.model.get("type_") == "mb-istft":
                    losses["subband_stft"] = loss_subband.item()
                LOG.info(
                    "Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(train_loader)
                    )
                )
                LOG.info(f"Losses: {losses}, step: {global_step}, lr: {lr}")

                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc_all,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                }
                scalar_dict.update(
                    {
                        "loss/g/fm": loss_fm,
                        "loss/g/mel": loss_mel,
                        "loss/g/kl": loss_kl,
                        "loss/g/lf0": loss_lf0,
                    }
                )
                if hps.model.get("type_") == "mb-istft":
                    scalar_dict["loss/g/subband"] = loss_subband

                # scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
                # scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
                # scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
                image_dict = {
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

                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict,
                )

            if global_step % hps.train.eval_interval == 0:
                LOG.info("Saving checkpoints...")
                _evaluate(hps, net_g, eval_loader, writer_eval)
                utils.save_checkpoint(
                    net_g,
                    optim_g,
                    hps.train.learning_rate,
                    epoch,
                    Path(hps.model_dir) / f"G_{global_step}.pth",
                )
                utils.save_checkpoint(
                    net_d,
                    optim_d,
                    hps.train.learning_rate,
                    epoch,
                    Path(hps.model_dir) / f"D_{global_step}.pth",
                )
                keep_ckpts = getattr(hps.train, "keep_ckpts", 0)
                if keep_ckpts > 0:
                    utils.clean_checkpoints(
                        path_to_models=hps.model_dir,
                        n_ckpts_to_keep=keep_ckpts,
                        sort_by_time=True,
                    )

        global_step += 1

    if rank == 0:
        global start_time
        now = time.time()
        durtaion = format(now - start_time, ".2f")
        LOG.info(f"====> Epoch: {epoch}, cost {durtaion} s")
        start_time = now


def _evaluate(
    hps: HParams,
    generator: torch.nn.Module,
    eval_loader: torch.utils.data.DataLoader,
    writer_eval: SummaryWriter,
) -> None:
    generator.eval()
    image_dict = {}
    audio_dict = {}
    with torch.no_grad():
        for batch_idx, items in enumerate(eval_loader):
            c, f0, spec, y, spk, _, uv = items
            g = spk[:1].cuda(0)
            spec, y = spec[:1].cuda(0), y[:1].cuda(0)
            c = c[:1].cuda(0)
            f0 = f0[:1].cuda(0)
            uv = uv[:1].cuda(0)
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_hat = generator.module.infer(c, f0, uv, g=g)

            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1).float(),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )

            audio_dict.update(
                {f"gen/audio_{batch_idx}": y_hat[0], f"gt/audio_{batch_idx}": y[0]}
            )
        image_dict.update(
            {
                "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy()),
                "gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy()),
            }
        )
    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate,
    )
    generator.train()
