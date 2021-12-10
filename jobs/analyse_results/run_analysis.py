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
print("Done with Imports")

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


def main(log_level: str = "DEBUG", display_only: bool = False):
    logging.basicConfig(
        level=LOG_LEVEL_NAMES[log_level],
        format="%(asctime)s %(levelname)s %(message)s\n",
    )
    
    analysis_dir = must_exist(SCRIPT_DIR / "retrieval_analysis_outputs")
    experiment_dir = must_exist(SCRIPT_DIR.parent / "retrieve_and_decode" / "iterated_decoding_output")
    target_path = must_exist(SCRIPT_DIR / "retrieval_analysis_lib.py")

    if not display_only:
        analysis_dir_names = set(path.name for path in analysis_dir.iterdir())
        targets = {
            path for path in experiment_dir.iterdir() 
            if path.name not in analysis_dir_names
        }
    else:
        targets = list(analysis_dir.iterdir())

        
    for path in targets:
        command = [sys.executable, str(target_path), path.name]
        if display_only:
            command.append("--display-only")
        command_str = shlex.join(command)  # Just for logging
        LOGGER.info(f"\nRunning:\n{command_str}")
        out = subprocess.run(command, capture_output=False, check=True)
        LOGGER.info(str(out) + "\n")


if __name__ == "__main__":
    fire.Fire(main)
