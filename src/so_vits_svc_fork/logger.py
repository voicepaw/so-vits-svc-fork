import os
from logging import (
    DEBUG,
    INFO,
    FileHandler,
    StreamHandler,
    basicConfig,
    captureWarnings,
)
from pathlib import Path

from rich.logging import RichHandler

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
    LOGGER_INIT = True
