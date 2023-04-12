import os
import sys
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

from rich.logging import RichHandler

LOGGER_INIT = False


def init_logger() -> None:
    global LOGGER_INIT
    if LOGGER_INIT:
        return

    IN_COLAB = os.getenv("COLAB_RELEASE_TAG")
    IS_TEST = "test" in Path.cwd().stem

    basicConfig(
        level=INFO,
        format="%(asctime)s %(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler() if not IN_COLAB else StreamHandler(),
            FileHandler(f"{__name__.split('.')[0]}.log"),
        ],
    )
    if IS_TEST:
        getLogger(sys.modules[__name__].__package__).setLevel(DEBUG)
    captureWarnings(True)
    LOGGER_INIT = True
