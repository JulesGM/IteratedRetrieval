#!/usr/bin/env bash
#SBATCH --job-name="gloo-10"
#SBATCH --gres=gpu:rtx8000:1
#SBATCH -c 16
#SBATCH --mem 48GB
#SBATCH -N 4
set -e

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

export PL_TORCH_DISTRIBUTED_BACKEND=gloo
ACTIVATION_PATH="$HOME/base_activate.sh"
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1

######################################################################
# Boilerplate to get the directory of the script.
######################################################################
if [[ -n "$SLURM_JOBID" ]] ; then
  #if we're in a slurm job:
  PATH=($(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}'))
  SCRIPT_DIR="$(dirname "${PATH[0]}")"
else
  SOURCE="${BASH_SOURCE[0]}"
  while [ -h "$SOURCE" ]; do 
    # resolve $SOURCE until the file is no longer a symlink
    DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" 
  done
  SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
fi

######################################################################
# Checking and Parsing Arguments.
# $1 Should be "conda" or "pip".
# $2 Should be the filename of the output directory.
# $3 Should be the path to the conda or pip (basic python) virtual environment
######################################################################
if [[ -z "$1" ]] ; then
  echo "\$1 needs to be either 'conda' or 'pip'."
  exit 1
fi

if [[ "$1" == "-h" || "$1" == "--help" ]] ; then
  echo \$1 Should be \"conda\" or \"pip\" according to the type of venv used
  echo \$2 Should be the filename of the output directory.
  echo \$3 Should be the path to the conda or pip \(basic python\) virtual environment.
fi

if [[ "$1" != "conda" && "$1" != "pip" ]] ; then 
  echo "\$1 needs to be either 'conda' or 'pip'."
  exit 1
fi

if [[ -z $2 ]] ; then
	echo "{$LOG_FRMT}\$2 is empty. Should be the filename of the output directory."
	exit 1;
fi

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
