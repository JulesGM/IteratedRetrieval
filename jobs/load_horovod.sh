#!/usr/bin/bash env

export NCCL_INCLUDE=/cvmfs/ai.mila.quebec/apps/x86_64/common/nccl/11.1-v2.8/include
export HOROVOD_NCCL_INCLUDE="$NCCL_INCLUDE"
# Not sure that this one does anything
export PATH="$NCCL_INCLUDE:$PATH"

export NCCL_LIB=/cvmfs/ai.mila.quebec/apps/x86_64/common/nccl/11.1-v2.8/lib
export HOROVOD_NCCL_LIB="$NCCL_LIB"
export LD_LIBRARY_PATH="$NCCL_LIB:$LD_LIBRARY_PATH"

export HOROVOD_GPU_OPERATIONS=NCCL 
export HOROVOD_WITH_PYTORCH=1

source "$HOME/condaless/bin/activate"