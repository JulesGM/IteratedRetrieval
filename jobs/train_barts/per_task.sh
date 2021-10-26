#!/usr/bin/env bash
set -e

echo "$0: ARGUMENTS:"
for arg in "$@" ; do
  echo - "\"$arg\""
done

# export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

##############################################################################
# Boilerplate to get the directory of the script.
##############################################################################
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do 
  # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" 
done
SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

##############################################################################
# Parse Args
##############################################################################
# shellcheck disable=SC1091
source "$SCRIPT_DIR/check_parse_args.sh" 

##############################################################################
# Behavior that changes according to the type of VENV.
##############################################################################
ACTIVATION_PATH="$ARG_ACTIVATION_PATH"
if [[ "$ARG_VENV_TYPE" == "pip" ]] ; then
  # shellcheck disable=SC1090
  source "$ACTIVATION_PATH"

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # Load the environment variables for Horovod and NCCL
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  module load openmpi
  module load cuda/11.1/nccl
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/load_horovod_vars.sh"

elif [[ "$ARG_VENV_TYPE" == "conda" ]] ;then  
  # shellcheck disable=SC1090 
  source "$ACTIVATION_PATH"
else
  echo "Bad \$ARG_VENV_TYPE: '$ARG_VENV_TYPE'"
  exit 1
fi

##############################################################################
# Setup that is the same for the two types of VENV
##############################################################################
ROOT="$(realpath "$SCRIPT_DIR/../../")"

if "$ARG_WITHOUT_CONTEXT" ; then
  echo "DOING WITHOUT CONTEXT"
  LAUNCHER_PATH="$ROOT/GAR/gar/our_launch_without_context.sh"
else
  echo "DOING WITH CONTEXT"
  LAUNCHER_PATH="$ROOT/GAR/gar/our_launch_with_context.sh"
fi

if [[ ! -e "$LAUNCHER_PATH" ]] ; then
    echo "${LOG_FRMT}FAILED: LAUNCHER_PATH does not exist."
    echo -e "$\tLAUNCHER_PATH: ${LAUNCHER_PATH}\n"
    exit 1
fi
LOG_FRMT="[$SLURM_NODEID,$SLURM_LOCALID] - "
GAR_PATH="$ROOT/GAR/"
cd "$GAR_PATH"

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL

echo "${LOG_FRMT}\$ACTIVATION_PATH: $ACTIVATION_PATH"
echo "${LOG_FRMT}NCCL_IB_DISABLE: ${NCCL_IB_DISABLE:-}"
echo "${LOG_FRMT}NCCL_P2P_DISABLE: ${NCCL_P2P_DISABLE:-}"
echo "${LOG_FRMT}PL_TORCH_DISTRIBUTED_BACKEND: ${PL_TORCH_DISTRIBUTED_BACKEND:-}"
echo "${LOG_FRMT}Which python: $(which python)"


##############################################################################
# Setup is over, just launch the script.
##############################################################################
# shellcheck disable=SC1090
source "$LAUNCHER_PATH" "$ARG_OUTPUT_FILENAME" "$ARG_DATASET_TYPE" "$ARG_BACKEND" 