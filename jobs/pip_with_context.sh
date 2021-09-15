#!/usr/bin/env bash
if [[ -z $1 ]] ; then
	echo "\$1 is empty. Should be the name of the output directory."
	exit 1;
fi

ROOT="/home/mila/g/gagnonju/IteratedDecoding"
ACTIVATION_PATH="$ROOT/condaless/bin/activate"
SCRIPT_PATH="$ROOT/GAR/gar/our_launch_with_context.sh"
LIBIVERBS_LOC="$ROOT/jobs/libiverbs"

module load python/3.8
source "$ACTIVATION_PATH"

module load openmpi
module load cuda/11.1/nccl
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export LD_LIBRARY_PATH="$LIBIVERBS_LOC:$LD_LIBRARY_PATH"

cd /home/mila/g/gagnonju/GAR/gar || exit

FRMT="[node #$SLURM_NNODES] - "
echo "${FRMT}NCCL_IB_DISABLE: $NCCL_IB_DISABLE"
echo "${FRMT}NCCL_P2P_DISABLE: $NCCL_P2P_DISABLE"
echo "${FRMT}PL_TORCH_DISTRIBUTED_BACKEND: $PL_TORCH_DISTRIBUTED_BACKEND"

source "$SCRIPT_PATH" "$1"
