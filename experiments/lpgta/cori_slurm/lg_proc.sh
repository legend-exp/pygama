#!/bin/bash
#SBATCH --chdir=/global/u1/l/lgprod/pygama/experiments/lpgta
#SBATCH --qos=shared
#SBATCH --time=12:00:00
#SBATCH --constraint=haswell
#SBATCH --account=m2676
#SBATCH --export=HDF5_USE_FILE_LOCKING=FALSE
#SBATCH --image=docker:legendexp/legend-base:latest
#SBATCH --output=/global/u1/l/lgprod/pygama/experiments/lpgta/cori_slurm/logs/cori-%j.txt

cd /global/u1/l/lgprod/pygama/experiments/lpgta
date
scontrol show job $SLURM_JOB_ID

# To list proc groups and their sizes, do 
# pdopenh5 ../LPGTA_fileDB.h5
# and then in python
# df.groupby(['proc_group'])['daq_size_GB'].sum()
# 
# To run all jobs do e.g.
# for i in {0..49}; do sbatch lg_proc.sh d2r $i; done
echo "-----------------------------------------------------------"
echo "processing $1 for group $2"
echo shifter python processing.py --dg --q "proc_group == $2" --$1 -o -v
shifter python processing.py --dg --q "proc_group == $2" --$1 -o -v
echo "-----------------------------------------------------------"

scontrol show job $SLURM_JOB_ID
date
