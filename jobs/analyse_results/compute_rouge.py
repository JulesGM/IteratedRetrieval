"""

"""
import concurrent.futures
from pathlib import Path
import os
import re
import shlex
import subprocess
import time

import colored_traceback.auto
import fire
import rich


SCRIPT_DIR = Path(__file__).absolute().parent

###############################################################################
# General Utils
###############################################################################
def check_exists(path):
    assert path.exists(), path
    return path


def check_all_exist(paths):
    outputs = []
    indices_dont_exist = []
    
    for i, path in enumerate(paths):
        outputs.append(path)
        if not path.exists():
            indices_dont_exist.append(i)

    assert not indices_dont_exist, [
        f"{i}: {indices_dont_exist[i]}" for i in indices_dont_exist
    ]

    return outputs


def check_is_truthy(obj):
    assert obj, obj
    return obj


def extract_number(path):
        matches = re.match(r"val_predictions-(\w+)(.*).txt", path.name)
        number = int(matches.group(1))
        return number


def model_output_paths_and_targets(directory):
    model_output_paths = check_is_truthy(
        check_all_exist(directory.glob("val_predictions-*.txt"))
    )
    model_output_paths.sort(key=extract_number, reverse=True)
    
    targets = check_exists(list(directory.glob("val_targets*.txt"))[0])
    return model_output_paths, targets


def prep_command(model_output_path, directory, targets):
    number = extract_number(model_output_path)
    assert isinstance(number, int), type(number)
    
    our_output = directory / f"rouge_{number}.txt"

    if not our_output.exists():
        rich.print(f"[yellow]Stacking {our_output}")
        command_flags = [
            f"--target_filepattern={shlex.quote(str(targets))}",
            f"--prediction_filepattern={shlex.quote(str(model_output_path))}",
            "--use_stemmer=True",
            f"--output_filename={shlex.quote(str(our_output))}",
        ]
        return command_flags, our_output
    return None, None
        

###############################################################################
# Thread specific
###############################################################################
def action_threads(our_output, command_flags):
    rich.print(f"[green]Starting {our_output}")

    full_cmd = [
        "python", "-m", "rouge_score.rouge"
    ] + command_flags

    rich.print(" ".join(full_cmd))
    subprocess.check_call(full_cmd)

    rich.print(f"[blue]Done with {our_output}")


def main(directory, max_threads):
    directory = Path(directory)
    assert directory.exists(), f"\nPath doesn't exist:\n{directory = }"
    model_output_paths, targets = model_output_paths_and_targets(directory)

    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_threads) as pool:
        for model_output_path in model_output_paths:
            maybe_command_flags, our_output = prep_command(
                model_output_path, directory, targets
            )
            if maybe_command_flags:
                pool.submit(
                    action_threads, 
                    our_output, 
                    maybe_command_flags,
                )
        for future in futures:
            future.wait()

    rich.print("[green bold]All done!")
               


if __name__ == "__main__":
    start = time.monotonic()
    fire.Fire(main)
    rich.print(f"[bold red]Total time: {time.monotonic() - start}")
