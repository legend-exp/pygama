#!/bin/bash
#
# With `env` distributions that implement the -S flag, it is possible to
# directly run the script inside the container with the following shebang:
#
#!/usr/bin/env -S singularity exec /path/to/legend-container.sif bash

root_dir="$(dirname "$(readlink -f "$0")")/../.."
prod_base="$root_dir/data/com"
run_name="sipmtest-202110"
meta_path="$root_dir/software/meta/com/$run_name"

export PYTHONUSERBASE="$root_dir/software/.local"

# "$root_dir/software/bin/pygama-run.py" \
#     --verbose \
#     --step daq_to_raw \
#     --input-files "$prod_base/daq/$run_name/.*\.fcio$" \
#     --output-dir "$prod_base/raw/$run_name" \
#     --config-file "$meta_path/d2r_config.json" # --overwrite

"$root_dir/software/bin/pygama-run.py" \
    --verbose \
    --step raw_to_dsp \
    --input-files "$prod_base/raw/$run_name/.*_raw\.lh5" \
    --output-dir "$prod_base/dsp/$run_name" \
    --config-file "$meta_path/r2d_config.json" # --overwrite
