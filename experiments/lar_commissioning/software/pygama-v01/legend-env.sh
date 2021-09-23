BASEDIR="/data1/shared/lar-commissioning"

VENV_NAME="legend-v01" SINGULARITYENV_PS1="\e[1m\][${VENV_NAME}]\[\e[m\] \w > " \
PYTHONUSERBASE="${BASEDIR}/software/pygama-v01/local" \
    \singularity shell -B /run/user/$(id -u) "${BASEDIR}/software/containers/legend-container.sif"
