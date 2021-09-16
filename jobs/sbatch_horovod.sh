#!/usr/bin/env bash
#SBATCH --job-name="horovod-16-rtx8000"
#SBATCH --gres=gpu:rtx8000:1
#SBATCH -c 16
#SBATCH --mem 48GB
#SBATCH -N 16
#SBATCH -n 1
set -e


SCRIPT_DIR=($(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}'))
SCRIPT_DIR="$(dirname "${SCRIPT_DIR[0]}")"
echo "$SCRIPT_DIR"

source "$HOME/condaless/bin/activate"
source "$SCRIPT_DIR/load_horovod.sh"

horovodrun -np 1 "$SCRIPT_DIR/with_context.sh" $@ horovod