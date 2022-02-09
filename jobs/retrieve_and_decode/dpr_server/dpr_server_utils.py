import contextlib
import io
import time

print("importing numpy")
import numpy as np
print("done importing numpy")
print("importing torch")
import rich
import torch
print("done importing torch")
import unary.unary_pb2_grpc as pb2_grpc
import unary.unary_pb2 as pb2


def encode_tensor(tensor, save_fn):
    with io.BytesIO() as f:
        save_fn(f, tensor)
        f.seek(0)
        return f.read()

def decode_tensor(bytes_, load_fn):
    with io.BytesIO() as f:
        f.write(bytes_)
        f.seek(0)
        tensor = load_fn(f)
    return tensor

def encode_torch_tensor(tensor):
    return encode_tensor(tensor, torch.save)

def decode_torch_tensor(tensor_proto):
    return decode_tensor(tensor_proto, torch.load)

def encode_numpy_tensor(tensor):
    return encode_tensor(tensor, np.save)

def decode_numpy_tensor(tensor_proto):
    return decode_tensor(tensor_proto, np.load)


@contextlib.contextmanager
def timeit(name):
    rich.print(
        f"[green bold]Starting timeit[/] -  [cyan bold]{name}[/]")
    start = time.monotonic()
    yield
    end = time.monotonic()
    rich.print(
        f"[green bold]Done with timeit[/] - [cyan bold]{name}[/]:"
        f" took [green bold]{end - start:0.1f}[/] s"
        )
