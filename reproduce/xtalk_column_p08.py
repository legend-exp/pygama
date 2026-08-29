"""Fill one p08 cross-talk column with `pygama.pargen.xtc.xtalk_column`.

One trigger detector per invocation, picked by index into the ``chn_id`` list
of ``test_p08.json``, so a SLURM array can cover all of them.

The per-channel baselines produced by ``prepare_baseline_p08.py`` are read back
from ``--baseline_dir`` and collected into the ``baseline`` dict
:func:`xtalk_column` expects.  They are collected in ``chn_id`` order, which is
what makes every column cover the same detectors in the same order -- the
condition :func:`~pygama.pargen.xtc.build_xtalk_matrix` checks later.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from pygama.pargen.xtc import xtalk_column

HERE = Path(__file__).resolve().parent
DEFAULT_FILE_LIST = HERE / "test_p08.json"
DEFAULT_OUT_PATH = Path("/pscratch/sd/h/hungwei/reproduce_temp/")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "trigger_index",
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
    help="Directory the per-trigger xtalk column lh5 is written to.",
)
parser.add_argument(
    "--baseline_dir",
    type=Path,
    default=None,
    help=(
        "Directory holding the baseline_XXXX.json files. "
        "Defaults to <out_path>/baseline."
    ),
)
parser.add_argument(
    "--debug_mode",
    action="store_true",
    help="Re-raise instead of falling back to an empty column or element.",
)
args = parser.parse_args()

# xtalk_column reports the skipped pairs and the failed elements through the
# logging module, so route those into the SLURM .out file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)

with args.file_list.open() as f:
    data = json.load(f)

hit_files = data["hit"]
dsp_files = data["dsp"]
chn_id = data["chn_id"]

if not 0 <= args.trigger_index < len(chn_id):
    msg = (
        f"trigger_index {args.trigger_index} out of range, "
        f"valid range is 0-{len(chn_id) - 1}"
    )
    raise ValueError(msg)

trigger = chn_id[args.trigger_index]
baseline_dir = args.baseline_dir or (args.out_path / "baseline")

# Collect the baselines in chn_id order: the keys of this dict become the
# response list of the column, so every column has to build it the same way.
# A channel whose baseline is missing stays in the dict with None values, which
# is what xtalk_column reads as "skip this element" while keeping the column
# the same length as every other one.
baseline = {}
missing = []
for index, detector in enumerate(chn_id):
    baseline_file = baseline_dir / f"baseline_{index:04d}.json"
    try:
        with baseline_file.open() as f:
            entry = json.load(f)
    except FileNotFoundError:
        missing.append(detector)
        baseline[detector] = {"positive_baseline": None, "negative_baseline": None}
        continue

    if entry["detector_id"] != detector:
        msg = (
            f"{baseline_file} holds detector {entry['detector_id']}, but index "
            f"{index} of {args.file_list} is detector {detector}"
        )
        raise ValueError(msg)

    baseline[detector] = entry

unusable = [
    detector
    for detector, entry in baseline.items()
    if entry["positive_baseline"] is None or entry["negative_baseline"] is None
]

print(
    f"trigger index {args.trigger_index} (ch{trigger}), "
    f"{len(hit_files)} hit files, {len(dsp_files)} dsp files, "
    f"{len(baseline)} response channels"
)
print(f"baselines read from: {baseline_dir}")
if missing:
    print(f"warning: no baseline file for {len(missing)} channels: {missing}")
if unusable:
    print(f"{len(unusable)} channels have no usable baseline: {unusable}")

# p08 ("xtc_old"): the trigger fires when the event is neither a discharge nor
# invalid, and the response side takes every coincident event.
config = {
    "trigger_conditions": {"is_discharge": False, "is_valid_0vbb_old": True},
    "response_conditions": {},
    "trigger_energy_range": (1500, 99999),
    "response_energy_range": (-99999, 100),
    "nbins": 700,
    "range_multiplier": 3,
}

out_file = args.out_path / "xtalk_columns" / f"xtalk_column_{args.trigger_index:04d}.lh5"

result = xtalk_column(
    hit_files=hit_files,
    dsp_files=dsp_files,
    trigger_detector_id=trigger,
    baseline=baseline,
    config=config,
    out_path=out_file,
    debug_mode=args.debug_mode,
)

n_valid = int(result["valid"].sum())
print(
    f"trigger ch{trigger}: {n_valid}/{len(result['response_ids'])} elements filled, "
    f"{int(result['n_events'].sum())} events over the whole column"
)
print(f"written to: {out_file}")
