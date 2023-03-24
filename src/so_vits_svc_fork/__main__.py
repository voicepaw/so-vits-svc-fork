from __future__ import annotations

import os
from logging import (
    DEBUG,
    INFO,
    FileHandler,
    StreamHandler,
    basicConfig,
    captureWarnings,
    getLogger,
)
from pathlib import Path
from typing import Literal

import click
import pyinputplus as pyip
import torch
from rich.logging import RichHandler

from so_vits_svc_fork import __version__

LOG = getLogger(__name__)
LOGGER_INIT = False


def init_logger() -> None:
    global LOGGER_INIT
    if LOGGER_INIT:
        return
    IN_COLAB = os.getenv("COLAB_RELEASE_TAG")
    IS_TEST = "test" in Path(__file__).parent.stem

    basicConfig(
        level=DEBUG if IS_TEST else INFO,
        format="%(asctime)s %(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler() if not IN_COLAB else StreamHandler(),
            FileHandler(f"{__name__.split('.')[0]}.log"),
        ],
    )
    captureWarnings(True)
    if IS_TEST:
        LOG.debug("Test mode is on.")

    LOG.info(f"Version: {__version__}")
    LOGGER_INIT = True


init_logger()


class RichHelpFormatter(click.HelpFormatter):
    def __init__(
        self,
        indent_increment: int = 2,
        width: int | None = None,
        max_width: int | None = None,
    ) -> None:
        width = 100
        super().__init__(indent_increment, width, max_width)


def patch_wrap_text():
    orig_wrap_text = click.formatting.wrap_text

    def wrap_text(
        text,
        width=78,
        initial_indent="",
        subsequent_indent="",
        preserve_paragraphs=False,
    ):
        return orig_wrap_text(
            text.replace("\n", "\n\n"),
            width=width,
            initial_indent=initial_indent,
            subsequent_indent=subsequent_indent,
            preserve_paragraphs=True,
        ).replace("\n\n", "\n")

    click.formatting.wrap_text = wrap_text


patch_wrap_text()

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"], show_default=True)
click.Context.formatter_class = RichHelpFormatter


@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    """so-vits-svc allows any folder structure for training data.
    However, the following folder structure is recommended.\n
        When training: dataset_raw/{speaker_name}/**/{wav_name}.{any_format}\n
        When inference: configs/44k/config.json, logs/44k/G_XXXX.pth\n
    If the folder structure is followed, you DO NOT NEED TO SPECIFY model path, config path, etc.
    (The latest model will be automatically loaded.)\n
    To train a model, run pre-resample, pre-config, pre-hubert, train.\n
    To infer a model, run infer.
    """
    init_logger()


@cli.command()
@click.option(
    "-c",
    "--config-path",
    type=click.Path(exists=True),
    help="path to config",
    default=Path("./configs/44k/config.json"),
)
@click.option(
    "-m",
    "--model-path",
    type=click.Path(),
    help="path to output dir",
    default=Path("./logs/44k"),
)
def train(config_path: Path, model_path: Path):
    """Train model
    If D_0.pth or G_0.pth not found, automatically download from hub."""
    from .train import train

    config_path = Path(config_path)
    model_path = Path(model_path)
    train(config_path=config_path, model_path=model_path)


@cli.command()
@click.argument(
    "input-path",
    type=click.Path(exists=True),
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(),
    help="path to output dir",
)
@click.option("-s", "--speaker", type=str, default=None, help="speaker name")
@click.option(
    "-m",
    "--model-path",
    type=click.Path(exists=True),
    default=Path("./logs/44k/"),
    help="path to model",
)
@click.option(
    "-c",
    "--config-path",
    type=click.Path(exists=True),
    default=Path("./configs/44k/config.json"),
    help="path to config",
)
@click.option(
    "-k",
    "--cluster-model-path",
    type=click.Path(exists=True),
    default=None,
    help="path to cluster model",
)
@click.option("-t", "--transpose", type=int, default=0, help="transpose")
@click.option(
    "-db", "--db-thresh", type=int, default=-20, help="threshold (DB) (RELATIVE)"
)
@click.option(
    "-a", "--auto-predict-f0", type=bool, default=True, help="auto predict f0"
)
@click.option(
    "-r", "--cluster-infer-ratio", type=float, default=0, help="cluster infer ratio"
)
@click.option("-n", "--noise-scale", type=float, default=0.4, help="noise scale")
@click.option("-p", "--pad-seconds", type=float, default=0.5, help="pad seconds")
@click.option(
    "-d",
    "--device",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="device",
)
@click.option("-ch", "--chunk-seconds", type=float, default=0.5, help="chunk seconds")
@click.option(
    "-ab", "--absolute-thresh", type=bool, default=False, help="absolute thresh"
)
def infer(
    input_path: Path,
    output_path: Path,
    speaker: str,
    model_path: Path,
    config_path: Path,
    cluster_model_path: Path | None = None,
    transpose: int = 0,
    db_thresh: int = -40,
    auto_predict_f0: bool = False,
    cluster_infer_ratio: float = 0,
    noise_scale: float = 0.4,
    pad_seconds: float = 0.5,
    chunk_seconds: float = 0.5,
    absolute_thresh: bool = False,
    device: Literal["cpu", "cuda"] = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Inference"""
    from .inference_main import infer

    if not auto_predict_f0:
        LOG.warning(
            f"auto_predict_f0 = False, transpose = {transpose}. If you want to change the pitch, please set transpose."
            "Generally transpose = 0 does not work because your voice pitch and target voice pitch are different."
        )

    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}.out{input_path.suffix}"
    output_path = Path(output_path)
    model_path = Path(model_path)
    if model_path.is_dir():
        model_path = list(sorted(model_path.glob("*.pth")))[-1]
        LOG.info(f"Since model_path is a directory, use {model_path}")
    config_path = Path(config_path)
    if cluster_model_path is not None:
        cluster_model_path = Path(cluster_model_path)
    infer(
        input_path=input_path,
        output_path=output_path,
        speaker=speaker,
        model_path=model_path,
        config_path=config_path,
        cluster_model_path=cluster_model_path,
        transpose=transpose,
        db_thresh=db_thresh,
        auto_predict_f0=auto_predict_f0,
        cluster_infer_ratio=cluster_infer_ratio,
        noise_scale=noise_scale,
        pad_seconds=pad_seconds,
        chunk_seconds=chunk_seconds,
        absolute_thresh=absolute_thresh,
        device=device,
    )


@cli.command()
@click.option(
    "-m",
    "--model-path",
    type=click.Path(exists=True),
    default=Path("./logs/44k/"),
    help="path to model",
)
@click.option(
    "-c",
    "--config-path",
    type=click.Path(exists=True),
    default=Path("./configs/44k/config.json"),
    help="path to config",
)
@click.option(
    "-k",
    "--cluster-model-path",
    type=click.Path(exists=True),
    default=None,
    help="path to cluster model",
)
@click.option("-t", "--transpose", type=int, default=12, help="transpose")
@click.option(
    "-a",
    "--auto-predict-f0",
    type=bool,
    default=False,
    help="auto predict f0 (not recommended for realtime since voice pitch will not be stable)",
)
@click.option(
    "-r", "--cluster-infer-ratio", type=float, default=0, help="cluster infer ratio"
)
@click.option("-n", "--noise-scale", type=float, default=0.4, help="noise scale")
@click.option(
    "-db", "--db-thresh", type=int, default=-30, help="threshold (DB) (ABSOLUTE)"
)
@click.option(
    "-fm",
    "--f0-method",
    type=click.Choice(["crepe", "crepe-tiny", "parselmouth", "dio", "harvest"]),
    default="dio",
    help="f0 prediction method",
)
@click.option("-p", "--pad-seconds", type=float, default=0.02, help="pad seconds")
@click.option("-ch", "--chunk-seconds", type=float, default=0.5, help="chunk seconds")
@click.option(
    "-cr",
    "--crossfade-seconds",
    type=float,
    default=0.01,
    help="crossfade seconds",
)
@click.option(
    "-ab",
    "--additional-infer-before-seconds",
    type=float,
    default=0.2,
    help="additional infer before seconds",
)
@click.option(
    "-aa",
    "--additional-infer-after-seconds",
    type=float,
    default=0.1,
    help="additional infer after seconds",
)
@click.option("-b", "--block-seconds", type=float, default=0.5, help="block seconds")
@click.option(
    "-d",
    "--device",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="device",
)
@click.option("-s", "--speaker", type=str, default=None, help="speaker name")
@click.option("-v", "--version", type=int, default=2, help="version")
@click.option("-i", "--input-device", type=int, default=None, help="input device")
@click.option("-o", "--output-device", type=int, default=None, help="output device")
@click.option(
    "-po",
    "--passthrough-original",
    type=bool,
    default=False,
    is_flag=True,
    help="passthrough original (for latency check)",
)
def vc(
    # paths
    model_path: Path,
    config_path: Path,
    # svc config
    speaker: str,
    cluster_model_path: Path | None,
    transpose: int,
    auto_predict_f0: bool,
    cluster_infer_ratio: float,
    noise_scale: float,
    f0_method: Literal["crepe", "crepe-tiny", "parselmouth", "dio", "harvest"],
    # slice config
    db_thresh: int,
    pad_seconds: float,
    chunk_seconds: float,
    # realtime config
    crossfade_seconds: float,
    additional_infer_before_seconds: float,
    additional_infer_after_seconds: float,
    block_seconds: float,
    version: int,
    input_device: int | str | None,
    output_device: int | str | None,
    device: Literal["cpu", "cuda"],
    passthrough_original: bool = False,
) -> None:
    """Realtime inference from microphone"""
    from .inference_main import realtime

    if auto_predict_f0:
        LOG.warning(
            "auto_predict_f0 = True in realtime inference will cause unstable voice pitch, use with caution"
        )
    else:
        LOG.warning(
            f"auto_predict_f0 = False, transpose = {transpose}. If you want to change the pitch, please change the transpose value."
            "Generally transpose = 0 does not work because your voice pitch and target voice pitch are different."
        )
    model_path = Path(model_path)
    config_path = Path(config_path)
    if cluster_model_path is not None:
        cluster_model_path = Path(cluster_model_path)
    if model_path.is_dir():
        model_path = list(sorted(model_path.glob("*.pth")))[-1]
        LOG.info(f"Since model_path is a directory, use {model_path}")

    realtime(
        # paths
        model_path=model_path,
        config_path=config_path,
        # svc config
        speaker=speaker,
        cluster_model_path=cluster_model_path,
        transpose=transpose,
        auto_predict_f0=auto_predict_f0,
        cluster_infer_ratio=cluster_infer_ratio,
        noise_scale=noise_scale,
        f0_method=f0_method,
        # slice config
        db_thresh=db_thresh,
        pad_seconds=pad_seconds,
        chunk_seconds=chunk_seconds,
        # realtime config
        crossfade_seconds=crossfade_seconds,
        additional_infer_before_seconds=additional_infer_before_seconds,
        additional_infer_after_seconds=additional_infer_after_seconds,
        block_seconds=block_seconds,
        version=version,
        input_device=input_device,
        output_device=output_device,
        device=device,
        passthrough_original=passthrough_original,
    )


@cli.command()
@click.option(
    "-i",
    "--input-dir",
    type=click.Path(exists=True),
    default=Path("./dataset_raw"),
    help="path to source dir",
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(),
    default=Path("./dataset/44k"),
    help="path to output dir",
)
@click.option("-s", "--sampling-rate", type=int, default=44100, help="sampling rate")
def pre_resample(input_dir: Path, output_dir: Path, sampling_rate: int) -> None:
    """Preprocessing part 1: resample"""
    from .preprocess_resample import preprocess_resample

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    preprocess_resample(
        input_dir=input_dir, output_dir=output_dir, sampling_rate=sampling_rate
    )


@cli.command()
@click.option(
    "-i",
    "--input-dir",
    type=click.Path(exists=True),
    default=Path("./dataset/44k"),
    help="path to source dir",
)
@click.option(
    "-f",
    "--filelist-path",
    type=click.Path(),
    default=Path("./filelists/44k"),
    help="path to filelist dir",
)
@click.option(
    "-c",
    "--config-path",
    type=click.Path(),
    default=Path("./configs/44k/config.json"),
    help="path to config",
)
def pre_config(
    input_dir: Path,
    filelist_path: Path,
    config_path: Path,
):
    """Preprocessing part 2: config"""
    from .preprocess_flist_config import preprocess_config

    input_dir = Path(input_dir)
    filelist_path = Path(filelist_path)
    config_path = Path(config_path)
    preprocess_config(
        input_dir=input_dir,
        train_list_path=filelist_path / "train.txt",
        val_list_path=filelist_path / "val.txt",
        test_list_path=filelist_path / "test.txt",
        config_path=config_path,
    )


@cli.command()
@click.option(
    "-i",
    "--input-dir",
    type=click.Path(exists=True),
    default=Path("./dataset/44k"),
    help="path to source dir",
)
@click.option(
    "-c",
    "--config-path",
    type=click.Path(exists=True),
    help="path to config",
    default=Path("./configs/44k/config.json"),
)
@click.option(
    "-n",
    "--n_jobs",
    "--n-jobs",
    type=int,
    default=4,
    help="number of jobs (optimal value may depend on your VRAM capacity and audio duration per file)",
)
@click.option(
    "-f",
    "--force-rebuild",
    type=bool,
    default=True,
    help="force rebuild existing preprocessed files",
)
@click.option(
    "-fm",
    "--f0-method",
    type=click.Choice(["crepe", "crepe-tiny", "parselmouth", "dio", "harvest"]),
    default="dio",
)
def pre_hubert(
    input_dir: Path,
    config_path: Path,
    n_jobs: bool,
    force_rebuild: bool,
    f0_method: Literal["crepe", "crepe-tiny", "parselmouth", "dio", "harvest"],
) -> None:
    """Preprocessing part 3: hubert
    If the HuBERT model is not found, it will be downloaded automatically."""
    from .preprocess_hubert_f0 import preprocess_hubert_f0

    input_dir = Path(input_dir)
    config_path = Path(config_path)
    preprocess_hubert_f0(
        input_dir=input_dir,
        config_path=config_path,
        n_jobs=n_jobs,
        force_rebuild=force_rebuild,
        f0_method=f0_method,
    )


@cli.command
def clean():
    """Clean up files, only useful if you are using the default file structure"""
    import shutil

    folders = ["dataset", "filelists", "logs"]
    if pyip.inputYesNo(f"Are you sure you want to delete files in {folders}?") == "yes":
        for folder in folders:
            if Path(folder).exists():
                shutil.rmtree(folder)
        LOG.info("Cleaned up files")
    else:
        LOG.info("Aborted")


@cli.command
@click.option(
    "-i",
    "--input-path",
    type=click.Path(exists=True),
    help="model path",
    default=Path("./logs/44k/"),
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(),
    help="onnx model path to save",
    default=None,
)
@click.option(
    "-c",
    "--config-path",
    type=click.Path(),
    help="config path",
    default=Path("./configs/44k/config.json"),
)
@click.option(
    "-d",
    "--device",
    type=str,
    default="cpu",
    help="device to use",
)
def onnx(input_path: Path, output_path: Path, config_path: Path, device: str) -> None:
    """Export model to onnx"""
    input_path = Path(input_path)
    if input_path.is_dir():
        input_path = list(input_path.glob("*.pth"))[0]
    if output_path is None:
        output_path = input_path.with_suffix(".onnx")
    output_path = Path(output_path)
    if output_path.is_dir():
        output_path = output_path / (input_path.stem + ".onnx")
    config_path = Path(config_path)
    device_ = torch.device(device)
    from .onnx_export import onnx_export

    onnx_export(
        input_path=input_path,
        output_path=output_path,
        config_path=config_path,
        device=device_,
    )


@cli.command
@click.option(
    "-i",
    "--input-dir",
    type=click.Path(exists=True),
    help="dataset directory",
    default=Path("./dataset/44k"),
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(),
    help="model path to save",
    default=Path("./logs/44k/kmeans.pt"),
)
@click.option("-n", "--n-clusters", type=int, help="number of clusters", default=2000)
@click.option(
    "-m/-nm", "--minibatch/--no-minibatch", default=True, help="use minibatch k-means"
)
def train_cluster(
    input_dir: Path, output_path: Path, n_clusters: int, minibatch: bool
) -> None:
    """Train k-means clustering"""
    from .cluster.train_cluster import main

    main(
        input_dir=input_dir,
        output_path=output_path,
        n_clusters=n_clusters,
        verbose=True,
        use_minibatch=minibatch,
    )
