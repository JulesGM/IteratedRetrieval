#!/usr/bin/env bash
#SBATCH --job-name="gloo-10"
#SBATCH --gres=gpu:rtx8000:1
#SBATCH -c 32
#SBATCH --mem 48GB
#SBATCH -N 2
#SBATCH -n 1
set -o errexit -o pipefail -o noclobber -o nounset


if [[ -z "$SLURM_JOBID" ]] ; then
    echo "Didn't get a job id. We are likely not in a Slurm task."
    exit 1
fi

###############################################################################
# Figure out the directory of the script
###############################################################################
# If the script was called by sbatch:
SLURM_COMMAND="$(scontrol show job "$SLURM_JOBID" | awk -F= '/Command=/{print $2}')"
IFS=" " read -r -a SCRIPT_ARGS <<< "$SLURM_COMMAND"
SCRIPT_DIR="$(dirname "${SCRIPT_ARGS[0]}")"

# If it was run in an interactive session:
if [[ "${SCRIPT_ARGS[*]}" == "(null)" ]] ; then
    ###################
    SOURCE="${BASH_SOURCE[0]}"
    while [ -h "$SOURCE" ]; do 
        # resolve $SOURCE until the file is no longer a symlink
        DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
        SOURCE="$(readlink "$SOURCE")"
        [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" 
    done
    SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
fi

###############################################################################
# Parse Arguments
###############################################################################
ARGS_COPY=("$@")
# shellcheck disable=SC1091
source "$SCRIPT_DIR/check_parse_args.sh" "$@" 

###############################################################################
# Logging stuff
###############################################################################
OUTPUT_FOLDER="$(realpath "${SCRIPT_DIR}/../GAR/gar/outputs/${ARG_OUTPUT_FILENAME}")"
TEE_TARGET="${OUTPUT_FOLDER}/console.txt"
OUTPUT_SAVE_PATH="${OUTPUT_FOLDER}/command_line_arguments.txt"

if [[ -e $OUTPUT_SAVE_PATH ]] ; then
    echo "Output path already exists: '$OUTPUT_SAVE_PATH'"
    echo "Choose something else or delete that folder."
fi

mkdir "$OUTPUT_FOLDER"
echo "${ARGS_COPY[@]}" > "$OUTPUT_SAVE_PATH"

###############################################################################
# Launch jobs
###############################################################################
if [[ $ARG_BACKEND == "horovod" ]] ; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/load_horovod_vars.sh"
    
    # spellcheck disable=SC1090
    source "$ARG_ACTIVATION_PATH"

    ###############################################################################
    # Populate the node list
    ###############################################################################
    # IFS="," read -r -a GPUS <<< "$SLURM_JOB_GPUS"
    # NGPUS="${#GPUS}"
    # NNODES="$SLURM_NNODES"
    # N_PROCESSES="$(("$NGPUS" * "$NNODES"))"
    # NODELIST="$SCRIPT_DIR/nodelist"
    # rm "${NODELIST}" || true
    # srun -l bash -c 'hostname' | sort -k 2 -u | awk -vORS=, '{print $2":4"}' \
    # | sed 's/,$//' > "$NODELIST"


    # ref: https://mit-satori.github.io/satori-workload-manager-using-slurm.html
    horovodrun -np "$SLURM_NTASKS" "$SCRIPT_DIR/per_task.sh" "${ARGS_COPY[@]}" 2>&1 | tee "$TEE_TARGET"
else
    srun --jobid "$SLURM_JOBID" "$SCRIPT_DIR/per_task.sh" "${ARGS_COPY[@]}" 2>&1 | tee "$TEE_TARGET"
fi
