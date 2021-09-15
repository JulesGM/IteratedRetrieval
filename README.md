Launch jobs with
```
srun --jobid $SLURM_JOB_ID jobs/conda_with_context.sh [output_path_name] [path_to_conda_venv_activate_file]
```

or with
```
srun --jobid $SLURM_JOB_ID jobs/pip_with_context.sh [output_path_name] [path_to_pip_venv_activate_file]
```

This assumes you are on a node of a multi-node job started with `salloc`.
