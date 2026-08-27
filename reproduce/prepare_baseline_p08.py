"""Measure the p08 baselines of one detector with `pygama.pargen.xtc.prepare_baseline`.

One detector per invocation, picked by index into the ``chn_id`` list of
``test_p08.json``, so a SLURM array can cover all of them.
"""

import argparse
import json
from pathlib import Path

from pygama.pargen.xtc import prepare_baseline

HERE = Path(__file__).resolve().parent
DEFAULT_FILE_LIST = HERE / "test_p08.json"
DEFAULT_OUT_PATH = Path("/pscratch/sd/h/hungwei/reproduce_temp/")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "detector_index",
    type=int,
    help="0-based index into the chn_id list of the file-list JSON.",
)
parser.add_argument(
    "--file_list",
    type=Path,
    default=DEFAULT_FILE_LIST,
    help="JSON holding the 'hit', 'dsp' and 'chn_id' lists.",
)
parser.add_argument(
    "--out_path",
    type=Path,
    default=DEFAULT_OUT_PATH,
    help="Directory the per-detector baseline JSON is written to.",
)
parser.add_argument(
    "--display",
    type=int,
    default=0,
    help="If > 0, write the before/after selection histograms next to the JSON.",
)
parser.add_argument(
    "--debug_mode",
    action="store_true",
    help="Re-raise instead of falling back to a nan result.",
)
args = parser.parse_args()

with args.file_list.open() as f:
    data = json.load(f)

hit_files = data["hit"]
dsp_files = data["dsp"]
chn_id = data["chn_id"]

if not 0 <= args.detector_index < len(chn_id):
    msg = (
        f"detector_index {args.detector_index} out of range, "
        f"valid range is 0-{len(chn_id) - 1}"
    )
    raise ValueError(msg)

detector = chn_id[args.detector_index]
print(
    f"detector index {args.detector_index} (ch{detector}), "
    f"{len(hit_files)} hit files, {len(dsp_files)} dsp files"
)

config = {"baseline_conditions": {"is_baseline": 63}}

out_file = args.out_path / "baseline" / f"baseline_{args.detector_index:04d}.json"

result = prepare_baseline(
    hit_files=hit_files,
    dsp_files=dsp_files,
    chn_id=detector,
    out_path=out_file,
    config=config,
    display=args.display,
    debug_mode=args.debug_mode,
)

print(f"result: {result}")
print(f"written to: {out_file}")
