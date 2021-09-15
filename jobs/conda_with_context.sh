#!/usr/bin/env bash

######################################################################
# Launches the script on the cluster interactively.
#
# Modify ACTIVATION_PATH with the location of your conda venv,
# or pass it as second argument.
#
#
# Call this script with 
# ```
# srun --jobid="$SLURM_JOB_ID" ./with_context.sh [output_folder_name] [[/path/to/conda_venv_activate.sh]]
# ```
######################################################################

ACTIVATION_PATH="$HOME/base_activate.sh"

######################################################################
# Boilerplate to get the directory of the script.
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do 
  # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" 
done
SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
######################################################################

source "$SCRIPT_DIR/shared.sh"
cd "$GAR_PATH" || exit

if [[ -z $1 ]] ; then
	echo "{$LOG_FRMT}\$1 is empty. Should be the name of the host node."
	exit 1;
fi

if [[ ! -z $2 ]] ; then
	ACTIVATION_PATH="$2"
fi
echo "${LOG_FRMT}\$ACTIVATION_PATH: $ACTIVATION_PATH"

# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
export PL_TORCH_DISTRIBUTED_BACKEND=gloo

echo "${LOG_FRMT}NCCL_IB_DISABLE: $NCCL_IB_DISABLE"
echo "${LOG_FRMT}NCCL_P2P_DISABLE: $NCCL_P2P_DISABLE"
echo "${LOG_FRMT}PL_TORCH_DISTRIBUTED_BACKEND: $PL_TORCH_DISTRIBUTED_BACKEND"

# shellcheck disable=SC1091
source "$ACTIVATION_PATH"
# shellcheck disable=SC1091
source "$LAUNCHER_PATH" "$1"
