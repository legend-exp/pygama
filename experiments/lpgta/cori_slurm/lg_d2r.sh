#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=5
#SBATCH --constraint=haswell
#SBATCH --account=m2676

cd ..
pwd
ls
whoami
shifter --image=docker:legendexp/legend-base:latest /bin/bash -c "python processing.py --dg"
srun shifter --image=docker:legendexp/legend-base:latest /bin/bash -c "python processing.py --dg"
#--q "YYYYmmdd == '20200728'" --d2r -o
