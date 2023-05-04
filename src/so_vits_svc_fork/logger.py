import os
import sys
from logging import DEBUG, INFO, StreamHandler, basicConfig, captureWarnings, getLogger
from pathlib import Path

from rich.logging import RichHandler

LOGGER_INIT = False


def init_logger() -> None:
    global LOGGER_INIT
    if LOGGER_INIT:
        return

    IS_TEST = "test" in Path.cwd().stem
    package_name = sys.modules[__name__].__package__
    basicConfig(
        level=INFO,
        format="%(asctime)s %(message)s",
        datefmt="[%X]",
        handlers=[
            StreamHandler() if is_notebook() else RichHandler(),
            # FileHandler(f"{package_name}.log"),
        ],
    )
    if IS_TEST:
        getLogger(package_name).setLevel(DEBUG)
    captureWarnings(True)
    LOGGER_INIT = True


def is_notebook():
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            raise ImportError("console")
            return False
        if "VSCODE_PID" in os.environ:  # pragma: no cover
            raise ImportError("vscode")
            return False
    except Exception:
        return False
    else:  # pragma: no cover
        return True
