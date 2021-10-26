print("(Re)/Loading iterated_utils.py")

import colorama
import contextlib
import json
import logging
from pathlib import Path
import time
import types
from typing import *

SCRIPT_DIR = Path(__file__).resolve().parent

import beartype
import colorama
import jsonlines
import numpy as np
import torch
import typeguard


try:
    import ujson as json
except ImportError:
    import json


LOGGER = logging.getLogger(__name__)
LOGGER.info(f"(Re)loaded {Path(__file__).name}")


@beartype.beartype
def add_to_err(err: Exception, new_content: str):
    assert len(err.args) == 1, len(err.args)
    err.args = (
        err.args[0] + "\n\n" +
        "Added information:\n" +
        new_content,
    )

    return err


def exception_title(text):
    return f"\n{colorama.Style.BRIGHT}{text}{colorama.Style.RESET_ALL}\n"



def check_init(cls):
    cls.__init__ = beartype.beartype(cls.__init__)


def check_equal(a, b):
    if not a == b:
        raise ValueError(
            exception_title("check_equal: a != b") +
            f"\t- {a = }\n"
            f"\t- {b = }"
        )


@beartype.beartype
def check_batch_size(tensor_size_0: int, batch_size: int, total_size: int) -> None:
    is_batch_size = tensor_size_0 == batch_size
    is_rest = tensor_size_0 == total_size % batch_size
    if not is_batch_size and not is_rest:
        message = (
            exception_title("check_batch_size failed:") +
            f"\t- {tensor_size_0 = }\n"
            f"\t- {batch_size = }\n"
            f"\t- {total_size = }\n"
            f"\t- {total_size % batch_size = }"
        )
        raise RuntimeError(message)


@beartype.beartype
def check_shape(checked_shape: Tuple[int, ...], objective_shape: Tuple[int, ...]) -> None:
    check_isinstance(checked_shape, tuple)

    if not tuple(checked_shape) == tuple(objective_shape):
        raise ValueError(
            exception_title("check_shape failed:") + 
            f"\t- checked_shape:   {checked_shape}, {np.prod(checked_shape)} elems\n"
            f"\t- objective_shape: {objective_shape}, {np.prod(objective_shape)} elems"
        )


def check_isinstance(obj: Any, type_: Type) -> None:
    assert isinstance(obj, type_), (
        f"Expected '{type_}', got {type(obj).mro()}"
    )


def _handle_paths(obj: Any) -> str:
    assert isinstance(obj, Path)
    return str(obj)


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return _handle_paths(obj)
    else:
        raise ValueError(type(obj).mro())
    check_isinstance(obj, iterated_retrieval.DecoderConf)

    
def format_dict(d: Dict[str, Any]) -> str:
    return json.dumps(
        d,
        indent=2,
        sort_keys=True,
        default=json_default,
    )


@contextlib.contextmanager
def time_this(title: str, no_start: bool = False) -> None:
    start = time.monotonic()
    bleu = colorama.Fore.BLUE
    green = colorama.Fore.GREEN
    reset = colorama.Style.RESET_ALL
    if not no_start:
        LOGGER.info(f"{bleu}Starting:{reset} {title}")
    yield "pizza"
    now = time.monotonic() - start
    LOGGER.info(f"{green}Done:{reset} {title}, {now:0.2f}s")


def save_json(obj: Any, path: Union[str, Path], *args, **kwargs):
    with open(path, "w") as fout:
        json.dump(obj, fout, *args, **kwargs)


def load_json(path: Union[str, Path], *args, **kwargs):
    with open(path) as fin:
        return json.load(fin, *args, **kwargs)


def class_checker(cls):
    for k, v in vars(cls).items():
        if isinstance(v, types.MethodType):
            vars(cls)[k] = beartype.beartype(cls[k])

    old_setattr = cls.__setattr__

    def checked_setattrib(self, attrib_name: str, new_value):
        hints = get_type_hints(self)[attrib_name]
        typeguard.check_type(attrib_name, new_value, hints)
        old_setattr(self, attrib_name, new_value)

    cls.__setattr__ = checked_setattrib

    return cls
