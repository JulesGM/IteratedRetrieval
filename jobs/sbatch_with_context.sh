#!/usr/bin/env bash
#SBATCH --job-name="gloo-10"
#SBATCH --gres=gpu:rtx8000:1
#SBATCH -c 16
#SBATCH --mem 48GB
#SBATCH -N 10
set -e


SCRIPT_DIR=($(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}'))
SCRIPT_DIR="$(dirname "${SCRIPT_DIR[0]}")"


# shellcheck disable=SC1091
source "$SCRIPT_DIR/check_args.sh" $@ 


srun --jobid "$SLURM_JOBID" "$SCRIPT_DIR/with_context.sh" $@
