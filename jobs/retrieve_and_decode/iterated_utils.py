print("(Re)/Loading iterated_utils.py")
import collections
import colorama
import contextlib
import enum
import json
import logging
import operator
from pathlib import Path
import time
import types
from typing import *

SCRIPT_DIR = Path(__file__).absolute().parent
LOGGER = logging.getLogger(__name__)

import beartype
import colored_traceback.auto
import colorama

import numpy as np
print("importing torch")
import torch
print("done importing torch")
import typeguard


try:
    import ujson as json
except ImportError:
    import json


def tensor_chunked(buf, chunk):
    for start in range(0, len(buf), chunk):
        yield buf[start:start + chunk]


class ValueEnumMeta(enum.EnumMeta):
    """__contains__ compares to the values. 
    Allows to do something like `"apple" in FruitEnum` instead of just 
    the equivalent of `FruitEnum.APPLE in FruitEnum`, assuming:
    
    ```
    class FruitEnum(enum.Enum):
        APPLE = "apple"
    ```

    This will be standard in Python 3.12.
    https://github.com/python/cpython/blob/main/Lib/enum.py#L621
    """
    def __contains__(cls, thing):
        return thing in cls._member_map_.values() or thing in cls._member_map_

    def error(cls, value):
            """This is just a super
            """
            print(dir(cls))
            if value in cls:
                raise NotImplementedError(
                    f"Valid value for enum <{cls.__name__}>, but pathway not implemented: {value}."
                )
            else:
                raise ValueError(
                    f"Invalid value for enum <{cls.__name__}>, got {value}, should be one of {list(cls)}."
                )


class ValueEnum(enum.Enum, metaclass=ValueEnumMeta):
    """__contains__ also compares to the values.
    This will be the standard in 3.12.
    https://github.com/python/cpython/blob/main/Lib/enum.py#L621
    """
    pass


class StrValueEnum(str, ValueEnum):
    pass


def logger_basicconfig():
    format_info = (
        "[%(levelname)s] (%(asctime)s) "
        "{%(name)s.%(funcName)s:%(lineno)d}:\n"
    )

    logging_format = (
        colorama.Fore.CYAN +
        format_info +
        colorama.Style.RESET_ALL +
        "%(message)s"
    )
    logging.basicConfig(
        format=logging_format,
        level=logging.WARNING,
        force=True,
    )


@beartype.beartype
def add_to_err(err: Exception, new_content: str):
    assert len(err.args) == 1, len(err.args)
    err.args = (
        err.args[0] + "\n\n" +
        "Added information:\n" +
        new_content,
    )

    return err


def check_all_equal(arr):
    if isinstance(arr, np.ndarray):
        assert arr.ndim == 1, arr.ndim
        assert len(arr), len(arr)

        return np.all(arr[0] == arr[1:])

    it_arr = iter(arr)
    try:
        first = next(it_arr)
    except StopIteration:
        raise ValueError("Requires at least one value or is undefined.")

    for other in it_arr:
        if first != other:
            return False
    return True


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
    elif hasattr(obj, "to_json") and callable(obj.to_json):
        return obj.to_json()
    else:
        raise ValueError(type(obj).mro())

    
def format_dict(d: Dict[str, Any]) -> str:
    return json.dumps(
        d,
        indent=2,
        sort_keys=True,
        default=json_default,
    )


def timestamp() -> str:
    return time.strftime("y%Ym%md%d-h%Hm%Ms%S")


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


def save_json(obj: Any, path: Union[str, Path], default: Optional[Callable] = None, *args, **kwargs):
    if default is None or default == json_default:
        default = json_default
    
    with open(path, "w") as fout:
        json.dump(obj, fout, default=default, *args, **kwargs)


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


# async def async_to_pdf(in_path, out_path):
#     in_path = str(in_path)
#     out_path = str(out_path)

#     print("awaiting launch")
#     browser = await launch()
#     print("Await newPage")
#     page = await browser.newPage()
#     print("Await goto")
#     await page.goto(f"file://{in_path}")
#     print("await screenshot")
#     await page.pdf({"path": out_path})
#     print("await close")
#     await browser.close()


# def to_pdf(in_path, out_path):
#     return asyncio.get_event_loop().run_until_complete(
#             async_to_pdf(
#                 in_path,
#                 out_path,
#             )
#         )


###############################################################################
# Advanced matrix operations
###############################################################################
def topk_w_torch(scores_np: np.ndarray, k, dim):
    assert scores_np.ndim == 2, scores_np.ndim
    other_dims = tuple([x for x in range(scores_np.ndim) if x != dim])

    scores_pt = torch.Tensor(scores_np)
    try:
        end = torch.topk(scores_pt, k=k, dim=dim).indices.numpy()
    except RuntimeError as err:
        raise add_to_err(
            err,
            f"{scores_pt.shape = }\n"
            f"{dim = }\n"
            f"{k = }\n"
        )

    if dim == 0:
        end = end.T

    check_shape(end.shape, tuple(scores.shape[x] for x in other_dims) + (k,))
    return end
    


def get_from_ith_dim(arr, dim, i):
    assert isinstance(i, (int, np.int64, np.int32)), i
    slice_ = tuple([... if d != dim else i for d in range(arr.ndim)])
    return arr[slice_]


def top_k_reference(scores: np.ndarray, k, dim):
    assert scores.ndim == 2, scores.ndim
    other_dim = abs(dim - 1)

    top_k = []
    for j in range(scores.shape[other_dim]):
        line = get_from_ith_dim(scores, other_dim, j)
        
        assert line.ndim == 1, line.ndim
        indices = np.arange(len(line))
        indices = sorted(indices, key=lambda i: line[i], reverse=True)[:k]
        top_k.append(
            indices
        )    

    top_k_np = np.array(top_k)
    check_shape(top_k_np.shape, (scores.shape[other_dim], k))    
    return top_k_np


class TopKJoinModes(StrValueEnum):
    MAX = "max"
    SUM = "sum"


def top_k_join(
    scores: np.ndarray, indices: np.ndarray, final_qty: int, mode: Union[TopKJoinModes, str]
):
    check_shape(scores.shape, indices.shape)
    check_equal(scores.ndim, 3)
    check_equal(indices.ndim, 3)

    output = []
    for batch_i in range(len(scores)):
        per_id = collections.defaultdict(int)   
        for query_i in range(len(scores[batch_i])):
            for retrieved_i in range(len(scores[batch_i][query_i])):
                index = indices[batch_i, query_i, retrieved_i]
                if mode == TopKJoinModes.MAX:
                    per_id[index] = max(per_id[index], scores[batch_i, query_i, retrieved_i])
                elif mode == TopKJoinModes.SUM:
                    per_id[index] += scores[batch_i, query_i, retrieved_i]
                else:
                    if mode in TopKJoinModes:
                        raise NotImplementedError(
                            f"Valid {mode = }, but not implemented. "
                            f"Try another of {list(TopKJoinModes)}."
                        )
                    else:
                        raise ValueError(f"{mode = } not in {list(TopKJoinModes)}.")

        top_k = sorted(per_id.items(), key=operator.itemgetter(1))[-final_qty:]

        top_k_keys = list(zip(*top_k))[0]
        output.append(top_k_keys)

    check_equal(len(output), scores.shape[0])
    output = np.asarray(output)
    check_shape(output.shape, (scores.shape[0], final_qty))

    return output


def top_k_sum(
    scores: np.ndarray, indices: np.ndarray, final_qty: int
):
    return top_k_join(scores, indices, final_qty, mode="sum")


def top_k_max(
    scores: np.ndarray, indices: np.ndarray, final_qty: int
):
    return top_k_join(scores, indices, final_qty, mode="max")


def get_torch(arr, indices):
    return torch.gather(
        input=torch.Tensor(arr), index=torch.Tensor(indices).long(), dim=1
    ).data.numpy()


def get_numpy(arr, indices):
    return np.take_along_axis(arr, indices, 1)


def get_reference(arr, indices):
    assert arr is not None
    assert indices is not None
    assert arr.shape[0] == indices.shape[0], (arr.shape[0], indices.shape[0])
    for batch_i in range(arr.shape[0]):
        arr[batch_i] = arr[batch_i, indices[batch_i]]
    return arr




if __name__ == "__main__":
    logger_basicconfig()
    
    # Test topk functions:
    # print("Starting the topk test")
    # for i in range(5):
    #     for j in range(2):
    #         scores = np.random.rand(6, 4)
    #         indices_pt = topk_w_torch(scores, k=3, dim=j)
    #         indices_reference = top_k_reference(scores, k=3, dim=j)
    #         cmp_pt = indices_pt == indices_reference
    #         assert np.all(cmp_pt), f"{np.mean(cmp_pt) = :0.1%}, {i = }, {j = }"
    # print("top_k passed")

    indices = np.array(
        [   
            [
                [0, 3, 2, 10], 
                [0, 1, 2, 3], 
                [0, 3, 2, 1],
            ],
            [
                [0, 1, 2, 3], 
                [0, 1, 2, 3], 
                [0, 14, 2, 3],
            ]
        ]
    )
    scores = np.array(
        [
            [
                [0.1, 0.2, 0.3, 0.7], 
                [0.1, 0.2, 0.3, 0.4], 
                [0.1, 0.9, 0.3, 0.4],
            ],
            [
                [0.4, 0.3, 0.2, 0.1], 
                [0.4, 0.3, 0.2, 0.1], 
                [0.4, 0.9, 0.2, 0.1],
            ]
        ]
    )
    print(f"{scores.shape = }")
    final_qty = 3
    indices_pt = top_k_sum(scores, indices, final_qty)
    print(f"{indices_pt.shape = }")
    print(indices_pt)
    print("done")


