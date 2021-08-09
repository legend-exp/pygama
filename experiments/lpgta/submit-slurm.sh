#!/bin/bash

# check the arguments
if [ "${1}" = "" ] || \
   [ "${2}" = "" ]; then
  echo "usage: ${0} [number of cores] [analysis script]"
  exit 1
fi

# create the output directory
NOW=$(date +"%F-%H-%M-%S")
mkdir ${SCRATCH}/${NOW}
echo "Created output area ${SCRATCH}/${NOW}"

# the file count
FILE_COUNT=`jq ".files_raw | .[]" database.json | wc -l`

# decrement the file count
((FILE_COUNT--))

# submit the jobs
sbatch \
    --account=m2676 \
    --array=0-${FILE_COUNT} \
    --chdir=${PWD} \
    --constraint=haswell \
    --cpus-per-task=${1} \
    --export=HDF5_USE_FILE_LOCKING=FALSE \
    --image=docker:legendexp/legend-software:latest \
    --licenses=SCRATCH \
    --nodes=1 \
    --ntasks=1 \
    --output=${SCRATCH}/${NOW}/slurm_%A.%a.out \
    --qos=shared \
    --time=01:00:00 \
    slurm.sh "${@}"
