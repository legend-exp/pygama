#!/bin/bash

BASEDIR="$(dirname "$(readlink -f "$0")")/.."

if [[ ! -f "$BASEDIR/bin/legend-container.sif" ]]; then
    if [[ -f /lfs/l1/legend/software/legend-base.sif ]]; then
        \ln -s /lfs/l1/legend/software/legend-base.sif "$BASEDIR/bin/legend-container.sif"
    else
        echo "Please copy or link a LEGEND container at $BASEDIR/bin/legend-container.sif"
        exit 1
    fi
fi

if [[ $# -eq 0 ]]; then
    VENV_NAME="legend-container" SINGULARITYENV_PS1="\e[1m\][${VENV_NAME}]\[\e[m\] \w > " \
    PYTHONUSERBASE="$BASEDIR/.local" \
        \singularity shell -B /run/user/$(id -u) "$BASEDIR/bin/legend-container.sif"
else
    VENV_NAME="legend-container" SINGULARITYENV_PS1="\e[1m\][${VENV_NAME}]\[\e[m\] \w > " \
    PYTHONUSERBASE="$BASEDIR/.local" \
        \singularity exec -B /run/user/$(id -u) "$BASEDIR/bin/legend-container.sif" $@
fi
