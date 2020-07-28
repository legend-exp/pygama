#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=5
#SBATCH --constraint=haswell
#SBATCH --account=m2676

export HDF5_USE_FILE_LOCKING=FALSE

echo "processing run $1"
cd ..
srun shifter --image=docker:legendexp/legend-base:latest python processing.py --dg --q "run == $1" --d2r -o -v
