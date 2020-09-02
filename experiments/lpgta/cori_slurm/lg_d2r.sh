#!/bin/bash
#SBATCH --chdir=/global/u1/l/lgprod/pygama/experiments/lpgta
#SBATCH --qos=shared
#SBATCH --time=4:00:00
#SBATCH --constraint=haswell
#SBATCH --account=m2676
#SBATCH --export=HDF5_USE_FILE_LOCKING=FALSE
#SBATCH --image=docker:legendexp/legend-base:latest
#SBATCH --output=/global/u1/l/lgprod/pygama/experiments/lpgta/cori_slurm/logs/cori-%j.txt

cd /global/u1/l/lgprod/pygama/experiments/lpgta
date

#echo "processing run $1"
#echo shifter python processing.py --dg --q "run == $1" --d2r -o -v
#shifter python processing.py --dg --q "run == $1" --d2r -o -v

echo "processing date $1"
echo shifter python processing.py --dg --date $1 --d2r -o -v
shifter python processing.py --dg --date $1 --d2r -o -v

#salloc -N 1 -C haswell -q interactive -t 04:00:00
#scontrol show job $JobID
