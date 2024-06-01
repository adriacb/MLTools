#!/bin/bash

ENV_PATH=$MLTOOLS_PREFIX'/etc/conda/activate.d/'
ENV_FILE=$ENV_PATH'env_vars.sh'
CONDA_BASE=$(conda info --base)
DIR="$(dirname "$(realpath "$0")")"

echo $DIR

mkdir -p $ENV_PATH
echo -n > $ENV_FILE

conda env config vars set MLTOOLS_PREFIX=$MLTOOLS_PREFIX

conda develop $MLTOOLS_PREFIX

# ADD BIN TO PATH
echo export PATH=$MLTOOLS_PREFIX'/bin':\$PATH >> $ENV_FILE