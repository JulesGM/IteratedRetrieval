#!/usr/bin/env bash
#SBATCH --job-name="gloo-10"
#SBATCH --gres=gpu:rtx8000:1
#SBATCH -c 16
#SBATCH --mem 48GB
#SBATCH -N 2
set -e

srun --jobid "$SLURM_JOBID" ./basic_test.py $@
