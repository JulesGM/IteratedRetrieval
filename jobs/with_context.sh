#!/usr/bin/env bash

if [[ -z $1 ]] ; then
	echo "\$1 is empty. Should be the name of the host node."
	exit 1;
fi

ROOT="/home/mila/g/gagnonju/IteratedDecoding"
ACTIVATION_PATH="$HOME/base_activate.sh"
SCRIPT_PATH="$ROOT/GAR/gar/our_launch_with_context.sh"
LIBIVERBS_LOC="$ROOT/jobs/libiverbs"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
# export PL_TORCH_DISTRIBUTED_BACKEND=gloo

export LD_LIBRARY_PATH="$LIBIVERBS_LOC:$LD_LIBRARY_PATH"

cd /home/mila/g/gagnonju/GAR/gar || exit

FRMT="[node #$SLURM_NNODES] - "
echo "${FRMT}NCCL_IB_DISABLE: $NCCL_IB_DISABLE"
echo "${FRMT}NCCL_P2P_DISABLE: $NCCL_P2P_DISABLE"
echo "${FRMT}PL_TORCH_DISTRIBUTED_BACKEND: $PL_TORCH_DISTRIBUTED_BACKEND"

# shellcheck disable=SC1091
source "$ACTIVATION_PATH"
# shellcheck disable=SC1091
source "$SCRIPT_PATH" "$1"
