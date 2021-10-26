#!/usr/bin/env bash
set -o errexit -o pipefail -o noclobber -o nounset

######################################################################
# Default values
######################################################################
CONDA_ACTIVATION_PATH_DEFAULT="$HOME/base_activate.sh"
PIP_ACTIVATION_PATH_DEFAULT="$HOME/condaless/bin/activate"
ARG_BACKEND_DEFAULT="horovod"
ARG_DEFAULT_ENV_TYPE="pip"


######################################################################
# Help
######################################################################
help () {
    echo \$1 Should be the dataset type, one of 'title', 'sentence' and 'answer'.
    echo \$2 Should be the filename of the output directory.
    echo "-t (optional) Should be \"conda\" or \"pip\" according to the type of venv used"
    echo "-p (optional) Should be the path to the conda or pip \(basic python\) virtual environment."
    echo "-b (optional) Should be the backend, one of 'horovod', 'nccl' and 'gloo'."
}

######################################################################
# Parse arguments https://stackoverflow.com/a/29754866/447599
######################################################################
! getopt --test > /dev/null 
if [[ ${PIPESTATUS[0]} -ne 4 ]]; then
    echo "I’m sorry, \`getopt --test\` failed in this environment."
    exit 1
fi

OPTIONS=ht:p:b:w
LONGOPTS=help,venv_type:,venv_path:,backend:,without_context

# -regarding ! and PIPESTATUS see above
# -temporarily store output to be able to check for errors
# -activate quoting/enhanced mode (e.g. by writing out “--options”)
# -pass arguments only via   -- "$@"   to separate them correctly
! PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTS --name "$0" -- "$@")
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    # e.g. return value is 1
    #  then getopt has complained about wrong arguments to stdout
    exit 2
fi
# read getopt’s output this way to handle the quoting right:
eval set -- "$PARSED"

######################################################################
# Init optional vars and assign unconditional defaults
######################################################################
ARG_BACKEND="$ARG_BACKEND_DEFAULT"
ARG_ACTIVATION_PATH=
ARG_VENV_TYPE="$ARG_DEFAULT_ENV_TYPE"
ARG_WITHOUT_CONTEXT=false

# Deal with args
while true; do
    case "$1" in
        -h|--help)
            help
            exit 0
            shift
            ;;
        -t|--venv_type)
            ARG_VENV_TYPE="$2"
            shift 2
            ;;
        -p|--venv_path)
            ARG_ACTIVATION_PATH="$2"
            shift 2
            ;;
        -b|--backend)
            ARG_BACKEND="$2"
            shift 2
            ;;
        -w|--without_context)
            ARG_WITHOUT_CONTEXT=true
            echo "WITHOUT CONTEXT"
            shift
            ;;
        --)
            shift
            break
            ;;
    esac
done

# Handle non-option arguments:
if [[ $# -ne 2 ]]; then
    echo $# "$@"
    echo "Two positional arguments are required, DATASET_TYPE and OUTPUT_FILENAME."
    exit 4
fi
# shellcheck disable=2034
ARG_DATASET_TYPE="$1"
# shellcheck disable=2034
ARG_OUTPUT_FILENAME="$2"


######################################################################
# Assign conditional defaults. 
######################################################################
if [[ -z "$ARG_ACTIVATION_PATH" && "$ARG_VENV_TYPE" == "conda" ]] ; then
    ARG_ACTIVATION_PATH="$CONDA_ACTIVATION_PATH_DEFAULT"
elif [[ -z "$ARG_ACTIVATION_PATH" && "$ARG_VENV_TYPE" == "pip" ]] ; then
    ARG_ACTIVATION_PATH="$PIP_ACTIVATION_PATH_DEFAULT"
fi


######################################################################
# Checking and Parsing Arguments.
# $1 Should be "conda" or "pip".
# $2 Should be the filename of the output directory.
# $3 Should be the path to the conda or pip (basic python) virtual environment

if [[ 
    "$ARG_VENV_TYPE" != "conda" && 
    "$ARG_VENV_TYPE" != "pip" 
    ]] ; then 
    echo "Error: \$ARG_VENV_TYPE needs to be either 'conda' or 'pip'."
    echo "-----------------------------------------------"
    help
	exit 1
fi

if [[ -z $2 ]] ; then
	echo "\$2 is empty. Should be the filename of the output directory."
    echo "-------------------------------------------------------------"
    help
	exit 1;
fi
if [[ 
    "$ARG_DATASET_TYPE" != "sentence" && 
    "$ARG_DATASET_TYPE" != "title" && 
    "$ARG_DATASET_TYPE" != "answer" 
    ]] ; then

    echo "'-t' Should be the type of inputs, one of 'sentence', 'title' or 'anser'. Got $ARG_DATASET_TYPE"
    echo "-------------------------------------------------------------"
    help
    exit 1

fi
if [[ 
    "$ARG_BACKEND" != "horovod" && 
    "$ARG_BACKEND" != "nccl" && 
    "$ARG_BACKEND" != "gloo" 
    ]] ; then
    echo "'\$4' Should be the backend, one of 'horovod', 'nccl' and 'gloo'. Got '$4'."
    echo "-------------------------------------------------------------"
    help
    exit 1
fi
if [[ ! -e "$ARG_ACTIVATION_PATH" ]] ; then
    echo "ARG_ACTIVATION_PATH does not exist: $ARG_ACTIVATION_PATH"
    echo "-------------------------------------------------------------"
    help
	exit 1
fi