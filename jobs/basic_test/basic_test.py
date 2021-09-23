#!/usr/bin/env python3
print("Started")
import os
import pathlib
import sys
global_rank = os.environ["SLURM_NODEID"]
PRE = f"#{global_rank} - "
print(f"{PRE}torch")
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
print(f"{PRE}torchvision")
import torchvision
print(f"{PRE}pytorch_lightning")
from pytorch_lightning.plugins.environments import slurm_environment
import pytorch_lightning.utilities.distributed
print(f"{PRE}Done")
from typing import *

def init_ddp_connection(
    cluster_environment: "pl.plugins.environments.ClusterEnvironment",
    torch_distributed_backend: str,
    global_rank: Optional[int] = None,
    world_size: Optional[int] = None,
    **kwargs: Any,
) -> None:
    # From https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/utilities/distributed.py#L363

    """Utility function to initialize DDP connection by setting env variables and initiliazing the distributed
    process group.
    Args:
        cluster_environment: ``ClusterEnvironment`` instance
        torch_distributed_backend: backend to use (includes `nccl` and `gloo`)
        global_rank: rank of the current process
        world_size: number of processes in the group
        kwargs: kwargs for ``init_process_group``
    """
    global_rank = global_rank if global_rank is not None else cluster_environment.global_rank()
    world_size = world_size if world_size is not None else cluster_environment.world_size()
    os.environ["MASTER_ADDR"] = cluster_environment.master_address()
    os.environ["MASTER_PORT"] = str(cluster_environment.master_port())
    if torch.distributed.is_available() and not torch.distributed.is_initialized():
        log.info(f"initializing ddp: GLOBAL_RANK: {global_rank}, MEMBER: {global_rank + 1}/{world_size}")
        torch.distributed.init_process_group(
            torch_distributed_backend, rank=global_rank, world_size=world_size, **kwargs
        )

        # on rank=0 let everyone know training is starting
        rank_zero_info(
            f"{'-' * 100}\n"
            f"distributed_backend={torch_distributed_backend}\n"
            f"All DDP processes registered. Starting ddp with {world_size} processes\n"
            f"{'-' * 100}\n"
        )

def main(env, global_rank, local_rank, num_replicas, backend):
    current_env = os.environ.copy()
    init_ddp_connection(env, backend)
    model = torchvision.models.mobilenet_v2()
    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    batch_size = 5 # Not important
    criterion = torch.nn.CrossEntropyLoss().cuda(local_rank)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    model = DDP(
        model, device_ids=[local_rank],
    )
    # Data loading code
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True,
        )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, 
        num_replicas=num_replicas, 
        rank=global_rank,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,         
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler,
    )  

    for x, y in train_loader:
        x = x.to(local_rank)
        y = y.to(local_rank)
        optimizer.zero_grad()
        
        pred = model(x)
        loss = criterion(pred, y)

        loss.backward()
        optimizer.step()


def display_info_env(env):
    env_info_keys = {
        "master_address",
        "master_port",
        "world_size",
        "global_rank",
        "local_rank",
        "node_rank",
        # "resolve_root_node_address",
    }
    env_info = dict()
    for key in env_info_keys:
        env_info[key] = getattr(env, key)()

    print(f"env_info: {env_info}")


if __name__ == "__main__":
    print("Started.")
    backend = "nccl"
    # This is what is used by Pytorch-Lightning to resolve the environment
    # https://github.com/PyTorchLightning/pytorch-lightning/blob/1.4.7/pytorch_lightning/plugins/environments/slurm_environment.py#L24
    # Called by https://github.com/PyTorchLightning/pytorch-lightning/blob/1.4.7/pytorch_lightning/trainer/connectors/accelerator_connector.py
    # Called by https://github.com/PyTorchLightning/pytorch-lightning/blob/1.4.7/pytorch_lightning/trainer/trainer.py#L354
    env = slurm_environment.SLURMEnvironment()
    display_info_env(env)
    main(
        env=env,
        global_rank=env.global_rank(), 
        local_rank=env.local_rank(),
        num_replicas=env.world_size(), 
        backend=backend,
    )