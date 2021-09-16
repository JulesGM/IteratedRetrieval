######################################################################
# Checking and Parsing Arguments.
# $1 Should be "conda" or "pip".
# $2 Should be the filename of the output directory.
# $3 Should be the path to the conda or pip (basic python) virtual environment

help () {
    echo \$1 Should be \"conda\" or \"pip\" according to the type of venv used
    echo \$2 Should be the filename of the output directory.
    echo \$3 Should be the dataset type, one of 'title', 'sentence' and 'answer'.
    echo \$4 Should be the backend, one of 'horovod', 'nccl' and 'gloo'.
    echo \$5 Should be the path to the conda or pip \(basic python\) virtual environment.
}

if [[ -z "$1" ]] ; then
    echo "Error: \$1 is empty."
    echo "--------------------"
    help
    exit 1
fi

if [[ "$1" == "-h" || "$1" == "--help" ]] ; then
    help
	exit 1
fi

if [[ "$1" != "conda" && "$1" != "pip" ]] ; then 
    echo "Error: \$1 needs to be either 'conda' or 'pip'."
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
if [[ "$3" != "sentence" && "$3" != "title" && "$3" != "answer" ]] ; then
    echo "'\$3' Should be the type of inputs, one of 'sentence', 'title' or 'anser'. Got $3"
    echo "-------------------------------------------------------------"
    help
    exit 1

fi
if [[ "$4" != "horovod" && "$4" != "nccl" && "$4" != "gloo" ]] ; then
    echo "'\$4' Should be the backend, one of 'horovod', 'nccl' and 'gloo'. Got '$4'."
    echo "-------------------------------------------------------------"
    help
    exit 1
fi
if [[ -z "$5" || ! -e "$5" ]] ; then
    echo "\$5 should be either empty or a file that exist. Got '$5'"
    echo "-------------------------------------------------------------"
    help
	exit 1
fi

######################################################################

ARG_VENV_TYPE="$1"
ARG_OUTPUT_FILENAME="$2"
ARG_DATASET_TYPE="$3"
ARG_BACKEND="$4"
ARG_ACTIVATION_PATH="$5"
