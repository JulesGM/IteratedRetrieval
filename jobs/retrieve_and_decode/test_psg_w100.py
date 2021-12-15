import collections
import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Union

try:
    import colored_traceback.auto
except ImportError:
    pass

import fire
import rich
import tqdm

DIR = Path(__file__).absolute().parent
DEFAULT_TARGET = (
    DIR / "downloads" / "data" / "wikipedia_split" / "psgs_w100.tsv"
)
PathType = Union[str, Path]


def count_num_lines(path: PathType) -> int:
    return int(subprocess.check_output(["wc", "-l", str(path)]).split()[0])


def print_counter(counter: collections.Counter) -> None:
    rich.print(sorted(counter.items(), key=lambda x: x[1]))


def calc_md5(path: PathType) -> str:
    return (
        subprocess.check_output(["md5sum", str(path)])
        .split()[0]
        .decode("utf-8")
    )


def main(path: PathType = DEFAULT_TARGET, N: int = 10000) -> None:
    path = Path(path)
    assert path.exists(), f"{path} does not exist"

    counter = collections.Counter()

    print("Counting lines...")
    num_lines = count_num_lines(path)
    print(f"{num_lines} lines")

    with open(path, "r") as f:
        reader = csv.reader(f, delimiter="\t")

        progress = tqdm.tqdm(enumerate(reader), total=num_lines)
        batch = []
        for i, line in progress:
            batch.append(len(line))
            if i % N == 0:
                counter.update(batch)
                batch = []
                progress.set_description(str(counter))
    print_counter(counter)


if __name__ == "__main__":
    fire.Fire(main)
