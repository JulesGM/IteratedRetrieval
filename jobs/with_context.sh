#!/usr/bin/env bash
set -e

echo "with_context.sh: ARGUMENTS: $@"

######################################################################
# Launches the script on the cluster interactively.
#
# Modify ACTIVATION_PATH with the location of your conda venv,
# or pass it as second argument.
#
#
# Call this script with 
# ```
# srun --jobid="$SLURM_JOB_ID" ./with_context.sh [[conda] or [pip]] [output_folder_name] [[/path/to/conda_venv_activate.sh]]
# ```
######################################################################

# export PL_TORCH_DISTRIBUTED_BACKEND=gloo
ACTIVATION_PATH="$HOME/base_activate.sh"
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1

######################################################################
# Boilerplate to get the directory of the script.
######################################################################

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do 
  # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" 
done
SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/check_args.sh" $@
if [[ -n $3 ]] ; then
	ACTIVATION_PATH="$3"
fi

######################################################################
# Behavior that changes according to the type of VENV.
######################################################################
if [[ "$1" == "pip" ]] ; then
  module load python/3.8

  # shellcheck disable=SC1090
  source "$ACTIVATION_PATH"

  module load openmpi
  module load cuda/11.1/nccl
else
  # shellcheck disable=SC1090 
  source "$ACTIVATION_PATH"
fi

######################################################################
# Setup that is the same for the two types of VENV
######################################################################
ROOT="$(realpath "$SCRIPT_DIR/../")"
LAUNCHER_PATH="$ROOT/GAR/gar/our_launch_with_context.sh"
LIBIVERBS_LOC="$ROOT/jobs/libiverbs"
GAR_PATH="$ROOT/GAR/"
LOG_FRMT="[node #$SLURM_NODEID] - "

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export LD_LIBRARY_PATH="$LIBIVERBS_LOC:$LD_LIBRARY_PATH"

echo "${LOG_FRMT}\$ACTIVATION_PATH: $ACTIVATION_PATH"
echo "${LOG_FRMT}NCCL_IB_DISABLE: $NCCL_IB_DISABLE"
echo "${LOG_FRMT}NCCL_P2P_DISABLE: $NCCL_P2P_DISABLE"
echo "${LOG_FRMT}PL_TORCH_DISTRIBUTED_BACKEND: $PL_TORCH_DISTRIBUTED_BACKEND"
echo "${LOG_FRMT}Which python: $(which python)"
echo "${LOG_FRMT}\$ACTIVATION_PATH: $ACTIVATION_PATH"

cd "$GAR_PATH"

######################################################################
# Setup is over, just launch the script.
######################################################################
# shellcheck disable=SC1090
source "$LAUNCHER_PATH" "$2"
