The default backend uses NVIDIA NCCL. 
There is another backend that uses Gloo (and also doesn't work.)
The backend can be set to `gloo` at the top of `jobs/with_context.sh`,
by uncommenting the appropriate line (PL_TORCH_DISTRIBUTED_BACKEND).

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

### 3. Launch the job

```
srun --jobid $SLURM_JOB_ID jobs/sbatch_with_context.sh [[pip] or [conda]] [output_path_name] [path_to_conda_venv_activate_file]
```

The first agrument after the script path can be either pip or conda according to the type of virtual environment being used.

