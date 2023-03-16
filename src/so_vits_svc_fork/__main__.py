from pathlib import Path
from typing import Literal

import click
import torch

from logging import getLogger
from logging import basicConfig, FileHandler, captureWarnings, INFO
from rich.logging import RichHandler

basicConfig(
    level=INFO,
    format="%(asctime)s %(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(), FileHandler(f"so-vits-svc-fork.log")],
)
captureWarnings(True)
LOG = getLogger(__name__)


@click.help_option("--help", "-h")
@click.group()
def cli():
    pass

    


@click.help_option("--help", "-h")
@cli.command()
@click.option("-c", "--config-path", type=click.Path(exists=True), help="path to config", default=Path("./configs/44k/config.json"))
@click.option("-m", "--model-path", type=click.Path(), help="path to output dir", default=Path("./logs/44k"))
def train(config_path: Path, model_path: Path):
    from .train import main

    main(config_path=config_path, model_path=model_path)


@click.help_option("--help", "-h")
@cli.command()
@click.option(
    "-i",
    "--input_path",
    type=click.Path(exists=True),
    help="path to source dir",
)
@click.option(
    "-o",
    "--output_path",
    type=click.Path(),
    help="path to output dir",
)
@click.option("-s", "--speaker", type=str, default=None, help="speaker name")
@click.option(
    "-m",
    "--model_path",
    type=click.Path(exists=True),
    default=Path("./logs/44k/G_800.pth"),
    help="path to model",
)
@click.option(
    "-c",
    "--config_path",
    type=click.Path(exists=True),
    default=Path("./configs/44k/config.json"),
    help="path to config",
)
@click.option(
    "-k",
    "--cluster_model_path",
    type=click.Path(exists=True),
    default=None,
    help="path to cluster model",
)
@click.option("-t", "--transpose", type=int, default=0, help="transpose")
@click.option("-d", "--db_thresh", type=int, default=-40, help="db thresh")
@click.option(
    "-a", "--auto_predict_f0", type=bool, default=False, help="auto predict f0"
)
@click.option(
    "-r", "--cluster_infer_ratio", type=float, default=0, help="cluster infer ratio"
)
@click.option("-n", "--noice_scale", type=float, default=0.4, help="noice scale")
@click.option("-p", "--pad_seconds", type=float, default=0.5, help="pad seconds")
@click.option(
    "-d",
    "--device",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="device",
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
    noice_scale: float = 0.4,
    pad_seconds: float = 0.5,
    device: Literal["cpu", "cuda"] = "cuda" if torch.cuda.is_available() else "cpu",
):
    from .inference_main import infer

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
        noice_scale=noice_scale,
        pad_seconds=pad_seconds,
        device=device,
    )


@click.help_option("--help", "-h")
@cli.command()
@click.option(
    "-i",
    "--input_dir",
    type=click.Path(exists=True),
    default=Path("./dataset_raw/44k"),
    help="path to source dir",
)
@click.option(
    "-o",
    "--output_dir",
    type=click.Path(),
    default=Path("./dataset/44k"),
    help="path to output dir",
)
@click.option("-s", "--sampling_rate", type=int, default=44100, help="sampling rate")
def preprocess(input_dir: Path, output_dir: Path, sampling_rate: int) -> None:
    from .preprocess_resample import preprocess_resample

    preprocess_resample(
        input_dir=input_dir, output_dir=output_dir, sampling_rate=sampling_rate
    )


@click.help_option("--help", "-h")
@cli.command()
@click.option(
    "-i",
    "--input_dir",
    type=click.Path(exists=True),
    default=Path("./dataset/44k"),
    help="path to source dir",
)
@click.option(
    "--filelist_path",
    type=click.Path(),
    default=Path("./filelists/44k"),
    help="path to filelist dir",
)
@click.option(
    "--config_path",
    type=click.Path(),
    default=Path("./configs/44k/config.json"),
    help="path to config",
)
def preprocess_config(
    input_dir: Path,
    filelist_path: Path,
    config_path: Path,
):
    from .preprocess_flist_config import preprocess_config

    preprocess_config(
        input_dir=input_dir,
        train_list_path=filelist_path / "train.txt",
        val_list_path=filelist_path / "val.txt",
        test_list_path=filelist_path / "test.txt",
        config_path=config_path,
    )


@click.help_option("--help", "-h")
@cli.command()
@click.option(
    "-i",
    "--input_dir",
    type=click.Path(exists=True),
    default=Path("./dataset/44k"),
    help="path to source dir",
)
@click.option(
    "-c",
    "--config_path",
    type=click.Path(exists=True),
    help="path to config",
    default=Path("./configs/44k/config.json"),
)
def preprocess_hubert(input_dir: Path, config_path: Path) -> None:
    from .preprocess_hubert_f0 import preprocess_hubert_f0

    preprocess_hubert_f0(input_dir=input_dir, config_path=config_path)
    
import pyinputplus as pyip
    
@cli.command
def clean():
    import shutil
    folders = ["dataset", "filelists", "logs"]
    if pyip.inputYesNo(f"Are you sure you want to delete files in {folders}?") == "yes":
        for folder in folders:
            shutil.rmtree(folder)
        LOG.info("Cleaned up files")
    else:
        LOG.info("Aborted")
