### 1. Pulling with Submodules

First pull with 

```
git clone git@github.com:JulesGM/IteratedRetrieval.git --recurse-submodules
```

to also download the **required** submodules.

### 2. Requirements

 - Install the requirements in `requirements.txt` in either a regular (pip) virtual environment or a conda one.

 - If you use conda, also install NCCL:
```
conda install -c conda-forge nccl
```

### 3. Get Nodes

Then nodes on slurm with something like
```
salloc --gres=gpu:rtx8000:1 -c 16 --mem 48GB -N 4
```

### 4. Interactively Start Script

Then Launch jobs with
```
srun --jobid $SLURM_JOB_ID jobs/conda_with_context.sh [output_path_name] [path_to_conda_venv_activate_file]
```

or with
```
srun --jobid $SLURM_JOB_ID jobs/pip_with_context.sh [output_path_name] [path_to_pip_venv_activate_file]
```
