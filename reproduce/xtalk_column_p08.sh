#!/bin/bash
#SBATCH --job-name=xtalk_column_p08
#SBATCH --output=/pscratch/sd/h/hungwei/reproduce_temp/logs/xtalk_column_p08_%A_%a.out
#SBATCH --error=/pscratch/sd/h/hungwei/reproduce_temp/logs/xtalk_column_p08_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-100%50
#SBATCH -q shared
#SBATCH -C cpu
#SBATCH -A m2676

# One array task per trigger detector: p08 has 101 channels, so 0-100.
# Each task loops over all 101 responses, so it does ~100x the reading a
# prepare_baseline task did -- hence the much longer wall clock.
#
# The log directory in the #SBATCH lines above is opened before this script
# runs, so it has to exist at submit time:
#   mkdir -p /pscratch/sd/h/hungwei/reproduce_temp/logs

OUT_PATH=${OUT_PATH:-/pscratch/sd/h/hungwei/reproduce_temp/}
FILE_LIST=${FILE_LIST:-reproduce/test_p08.json}
BASELINE_DIR=${BASELINE_DIR:-/pscratch/sd/h/hungwei/reproduce_temp/baseline}

# Always work from the repository root
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(readlink -f "$0")")")}
cd "$REPO_ROOT" || exit 1

source .venv/bin/activate

date
hostname
echo "Running xtalk_column on trigger ${SLURM_ARRAY_TASK_ID}, file list ${FILE_LIST}, baselines from ${BASELINE_DIR}, results in ${OUT_PATH}"

python reproduce/xtalk_column_p08.py \
    --file_list "${FILE_LIST}" \
    --out_path "${OUT_PATH}" \
    --baseline_dir "${BASELINE_DIR}" \
    "${SLURM_ARRAY_TASK_ID}"

echo "Done."
date
