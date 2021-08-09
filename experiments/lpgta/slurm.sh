#!/bin/bash

# check the arguments
if [ "${1}" = "" ] || \
   [ "${2}" = "" ]; then
  echo "usage: ${0} [number of cores] [analysis script]"
  exit 1
fi

# loop over the arguments
for ARG in "${@:2}"; do
    # the command to execute
    CMD="srun --cpu_bind=cores shifter python ${ARG} -c ${1} -i ${SLURM_ARRAY_TASK_ID}"
    
    # status printout
    echo -e "`date`\n"
    echo -e "${CMD}\n"
    
    # execute the command
    ${CMD}
done

# status printout
echo -e "\n`date`"
