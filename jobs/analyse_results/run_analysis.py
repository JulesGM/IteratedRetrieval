import logging
import os
from pathlib import Path
from typing import *
import shlex
import subprocess
import sys

import fire
try:
    import colored_traceback.auto
except ImportError:
    pass

SCRIPT_DIR = Path(__file__).absolute().parent
LOGGER = logging.getLogger(__name__)
LOG_LEVEL_NAMES = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}


PathType = Union[str, Path]


def must_exist(path: PathType) -> Path:
    path = Path(path)
    if not path.exists():
        raise ValueError(f"{path} does not exist")
    return path


def main(log_level: str = "DEBUG"):
    logging.basicConfig(
        level=LOG_LEVEL_NAMES[log_level],
        format="%(asctime)s %(levelname)s %(message)s\n",
    )
    analysis_dir = must_exist(SCRIPT_DIR / "retrieval_analysis_outputs")
    experiment_dir = must_exist(SCRIPT_DIR.parent / "retrieve_and_decode")
    target_path = must_exist(SCRIPT_DIR / "retrieval_analysis_lib.py")


    exp_dir_names = set(path.name for path in experiment_dir.iterdir())
    not_done_yet = {
        path for path in analysis_dir.iterdir() 
        if path.name not in exp_dir_names
    }
    
    for path in not_done_yet:
        command = [sys.executable, str(target_path), str(path)]
        command_str = shlex.join(command)
        LOGGER.info(f"\nRunning:\n{command_str}")
        out = subprocess.run(command, check=True)
        LOGGER.info(out + "\n")


if __name__ == "__main__":
    fire.Fire(main)