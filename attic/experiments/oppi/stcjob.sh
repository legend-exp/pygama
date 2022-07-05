#!/bin/bash
#SBATCH --qos=shared
#SBATCH --time=24:00:00
#SBATCH --constraint=haswell
#SBATCH --account=m2676
#SBATCH --export=HDF5_USE_FILE_LOCKING=FALSE
#SBATCH --image=legendexp/legend-base:latest
#SBATCH --chdir=/global/homes/w/wisecg/pygama_wisecg/experiments/oppi
#SBATCH --output=/global/homes/w/wisecg/pygama_wisecg/experiments/oppi/logs/cori-%j.txt
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=wisecg@uw.edu

echo "Job Start:"
date
echo "Node(s):  "$SLURM_JOB_NODELIST
echo "Job ID:  "$SLURM_JOB_ID

if [ -n "$SHIFTER_RUNTIME" ]; then
  echo "Shifter image active."
  echo "pwd: "`pwd`
  echo "gcc: "$CC
  echo "g++:"$CXX
  echo "Python:"`python --version`
  echo "ROOT:"`root-config --version`
fi

# NOTE, don't run -v on --d2r jobs, there's a progress bar that fills up
# log files with garbage

# run quick test (debug, 30 min)
# shifter python processing.py -q 'cycle==2019' --d2r --r2d -o

# overwrite everything (~24 hr job, shared queue)
shifter python processing.py -q 'cycle>=0' --d2r --r2d -o

# update everything
# shifter python processing.py -q 'cycle>0' --d2r --r2d

# This runs whatever we pass to it (maybe from python)
# echo "${@}"
# ${@}

echo "Job Complete:"
date