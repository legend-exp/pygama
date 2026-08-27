#!/bin/bash
#SBATCH --job-name=baseline_p08
#SBATCH --output=reproduce/logs/baseline_p08_%A_%a.out
#SBATCH --error=reproduce/logs/baseline_p08_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-100%50
#SBATCH -q shared
#SBATCH -C cpu
#SBATCH -A m2676

# One array task per detector: p08 has 101 channels, so 0-100.

OUT_PATH=${OUT_PATH:-/pscratch/sd/h/hungwei/reproduce_temp/}
FILE_LIST=${FILE_LIST:-reproduce/test_p08.json}
DISPLAY_LEVEL=${DISPLAY_LEVEL:-0}

# Always work from the repository root
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(readlink -f "$0")")")}
cd "$REPO_ROOT" || exit 1

source .venv/bin/activate

mkdir -p reproduce/logs

date
hostname
echo "Running prepare_baseline on detector ${SLURM_ARRAY_TASK_ID}, file list ${FILE_LIST}, results in ${OUT_PATH}"

python reproduce/prepare_baseline_p08.py \
    --file_list "${FILE_LIST}" \
    --out_path "${OUT_PATH}" \
    --display "${DISPLAY_LEVEL}" \
    "${SLURM_ARRAY_TASK_ID}"

echo "Done."
date
