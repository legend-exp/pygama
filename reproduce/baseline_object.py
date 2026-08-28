"""Collect per-detector baseline JSON files into the ``baseline`` argument of
:func:`pygama.pargen.xtc.xtalk_column`, and compare two such collections.

``prepare_baseline`` writes one JSON file per detector; ``xtalk_column`` wants
them all in a single mapping.  :func:`load_baseline` does that join.  The rest
of the module exists to answer a narrower question: do two independently
produced sets of baseline files describe the *same* measurement?  They will
never be byte-identical -- they carry different provenance fields and were
printed by different writers -- so :func:`compare_baselines` reports the
difference at the level that matters: which detectors each side covers, which
of them are usable, and how far apart the numbers actually are, measured both
as a relative difference and in float32 ULPs.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

BASELINE_KEYS = ("detector_id", "positive_baseline", "negative_baseline")
VALUE_KEYS = ("positive_baseline", "negative_baseline")

DEFAULT_DIR_A = Path("/pscratch/sd/h/hungwei/reproduce_temp/baseline")
DEFAULT_DIR_B = Path(
    "/pscratch/sd/h/hungwei/temp_results_p08/temp_results/parameters"
    "/baseline_individuals_20260719_234907/json"
)


# ----------------------------------------------------------------------------
# building the baseline object
# ----------------------------------------------------------------------------


def load_baseline(
    json_files: Iterable[str | Path],
    strict: bool = True,
) -> dict:
    """Join per-detector baseline JSON files into one ``baseline`` mapping.

    Parameters
    ----------
    json_files
        Paths of the JSON files written by
        :func:`pygama.pargen.xtc.prepare_baseline`.  Each must be an object
        holding ``detector_id``, ``positive_baseline`` and
        ``negative_baseline``; any other key it carries is ignored.
    strict
        If True, a file that cannot be parsed, is missing one of those three
        keys, or repeats a ``detector_id`` already seen raises ``ValueError``.
        If False, the file is logged and skipped instead (a repeated
        ``detector_id`` keeps the first file's values).

    Returns
    -------
    dict
        ``{detector_id: {"positive_baseline": float | None,
        "negative_baseline": float | None}}``, ready to pass as the
        ``baseline`` argument of :func:`pygama.pargen.xtc.xtalk_column`.
        Detectors whose measurement failed keep their ``None`` values --
        ``xtalk_column`` treats those as unusable and skips them, so dropping
        them here would only hide them.
    """
    baseline = {}
    seen_in = {}

    for path in json_files:
        path = Path(path)

        try:
            with path.open() as f:
                record = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            msg = f"{path}: cannot be read as JSON: {type(e).__name__}: {e}"
            if strict:
                raise ValueError(msg) from e
            log.warning("%s, skipping", msg)
            continue

        if not isinstance(record, dict):
            msg = f"{path}: holds a {type(record).__name__}, not a JSON object"
            if strict:
                raise ValueError(msg)
            log.warning("%s, skipping", msg)
            continue

        missing = [key for key in BASELINE_KEYS if key not in record]
        if missing:
            msg = f"{path}: missing key(s) {', '.join(missing)}"
            if strict:
                raise ValueError(msg)
            log.warning("%s, skipping", msg)
            continue

        detector_id = record["detector_id"]
        if detector_id in seen_in:
            msg = (
                f"{path}: detector_id {detector_id} already taken from "
                f"{seen_in[detector_id]}"
            )
            if strict:
                raise ValueError(msg)
            log.warning("%s, keeping the first", msg)
            continue

        seen_in[detector_id] = path
        baseline[detector_id] = {
            "positive_baseline": record["positive_baseline"],
            "negative_baseline": record["negative_baseline"],
        }

    return baseline


def baseline_files(directory: str | Path, pattern: str = "*.json") -> list[Path]:
    """Every JSON file in *directory*, sorted by name."""
    return sorted(Path(directory).glob(pattern))


def load_baseline_dir(
    directory: str | Path, pattern: str = "*.json", strict: bool = True
) -> dict:
    """:func:`load_baseline` over every JSON file in *directory*."""
    return load_baseline(baseline_files(directory, pattern), strict=strict)


# ----------------------------------------------------------------------------
# describing the difference between two baseline objects
# ----------------------------------------------------------------------------


def _usable(value) -> bool:
    """Whether *value* is a number ``xtalk_column`` would actually use."""
    if value is None:
        return False
    try:
        return not math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _float32_rank(x: float) -> int:
    """Index of ``float32(x)`` in the ascending order of all float32 values.

    Adjacent float32 values differ by a rank of one, so the difference of two
    ranks counts the representable numbers between them -- the ULP distance.
    """
    bits = int(np.float32(x).view(np.uint32))
    magnitude = bits & 0x7FFFFFFF
    return -magnitude if bits & 0x80000000 else magnitude


def float32_ulps(a: float, b: float) -> int:
    """How many float32 values lie between *a* and *b* once both are rounded."""
    return abs(_float32_rank(a) - _float32_rank(b))


def relative_difference(a: float, b: float) -> float:
    """``|a - b|`` over the larger magnitude; 0.0 when both are zero."""
    scale = max(abs(a), abs(b))
    if scale == 0.0:
        return 0.0
    return abs(a - b) / scale


def agreeing_digits(a: float, b: float) -> float:
    """Leading significant decimal digits *a* and *b* have in common.

    ``inf`` when they are bit-for-bit equal.
    """
    rel = relative_difference(a, b)
    return math.inf if rel == 0.0 else -math.log10(rel)


def compare_baselines(baseline_a: dict, baseline_b: dict) -> dict:
    """Describe how two baseline objects differ.

    Coverage and usability are compared first, because a detector present in
    only one of the two, or usable in only one of the two, is a difference no
    numerical tolerance can express.  The numbers themselves are then compared
    only over the detectors both sides can actually use.

    Returns
    -------
    dict
        ``only_in_a``/``only_in_b``/``common`` detector id lists,
        ``usable_in_a_only``/``usable_in_b_only``/``usable_in_both``, and
        ``values``: one record per (detector, field) pair compared, each with
        ``a``, ``b``, ``abs_diff``, ``rel_diff``, ``ulps`` (float32) and
        ``digits``.
    """
    ids_a = set(baseline_a)
    ids_b = set(baseline_b)
    common = sorted(ids_a & ids_b, key=str)

    usable_in_both = []
    usable_in_a_only = []
    usable_in_b_only = []
    values = []

    for detector_id in common:
        entry_a = baseline_a[detector_id]
        entry_b = baseline_b[detector_id]

        ok_a = all(_usable(entry_a.get(key)) for key in VALUE_KEYS)
        ok_b = all(_usable(entry_b.get(key)) for key in VALUE_KEYS)

        if ok_a and not ok_b:
            usable_in_a_only.append(detector_id)
            continue
        if ok_b and not ok_a:
            usable_in_b_only.append(detector_id)
            continue
        if not (ok_a or ok_b):
            continue

        usable_in_both.append(detector_id)
        for key in VALUE_KEYS:
            a = float(entry_a[key])
            b = float(entry_b[key])
            values.append(
                {
                    "detector_id": detector_id,
                    "field": key,
                    "a": a,
                    "b": b,
                    "abs_diff": abs(a - b),
                    "rel_diff": relative_difference(a, b),
                    "ulps": float32_ulps(a, b),
                    "digits": agreeing_digits(a, b),
                }
            )

    return {
        "only_in_a": sorted(ids_a - ids_b, key=str),
        "only_in_b": sorted(ids_b - ids_a, key=str),
        "common": common,
        "usable_in_both": usable_in_both,
        "usable_in_a_only": usable_in_a_only,
        "usable_in_b_only": usable_in_b_only,
        "values": values,
    }


# ----------------------------------------------------------------------------
# how many digits each source actually wrote
# ----------------------------------------------------------------------------

_NUMBER = re.compile(
    r'"(positive_baseline|negative_baseline)"\s*:\s*(-?\d+\.?\d*(?:[eE][-+]?\d+)?)'
)


def printed_digits(json_files: Iterable[str | Path]) -> list[int]:
    """Significant decimal digits of every baseline literal, as written.

    Read off the file text rather than the parsed floats: a float carries no
    memory of how it was printed, and "did the two sources write the same
    number of digits" is a question about the text.
    """
    counts = []
    for path in json_files:
        try:
            text = Path(path).read_text()
        except OSError as e:
            log.warning("%s: cannot be read: %s, skipping", path, e)
            continue
        for _, literal in _NUMBER.findall(text):
            mantissa = literal.split("e")[0].split("E")[0]
            digits = mantissa.lstrip("-").replace(".", "").lstrip("0")
            counts.append(len(digits))
    return counts


# ----------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------


def _summarise(name: str, xs: list[float]) -> str:
    if not xs:
        return f"  {name}: no values"
    xs = sorted(xs)
    return (
        f"  {name}: min {xs[0]:.4g}, median {xs[len(xs) // 2]:.4g}, "
        f"max {xs[-1]:.4g}"
    )


def _id_list(ids: list, limit: int = 12) -> str:
    if not ids:
        return "none"
    shown = ", ".join(str(i) for i in ids[:limit])
    return shown if len(ids) <= limit else f"{shown}, ... ({len(ids)} total)"


def format_report(
    report: dict,
    label_a: str,
    label_b: str,
    digits_a: list[int] | None = None,
    digits_b: list[int] | None = None,
    n_worst: int = 5,
) -> str:
    """Render :func:`compare_baselines` output as a readable block of text."""
    lines = [
        "=" * 78,
        f"A = {label_a}",
        f"B = {label_b}",
        "=" * 78,
        "",
        "coverage",
        f"  in both:    {len(report['common'])} detectors",
        f"  only in A:  {_id_list(report['only_in_a'])}",
        f"  only in B:  {_id_list(report['only_in_b'])}",
        "",
        "usability (both baselines present and not NaN)",
        f"  usable in both:   {len(report['usable_in_both'])}",
        f"  usable in A only: {_id_list(report['usable_in_a_only'])}",
        f"  usable in B only: {_id_list(report['usable_in_b_only'])}",
        "",
    ]

    values = report["values"]
    if not values:
        lines.append("no detector is usable on both sides, nothing to compare")
        return "\n".join(lines)

    identical = [v for v in values if v["abs_diff"] == 0.0]
    within_1_ulp = [v for v in values if v["ulps"] <= 1]
    lines += [
        f"numeric agreement over {len(values)} values "
        f"({len(report['usable_in_both'])} detectors x {len(VALUE_KEYS)} fields)",
        f"  bit-for-bit identical:       {len(identical)} / {len(values)}",
        f"  within 1 float32 ULP:        {len(within_1_ulp)} / {len(values)}",
        _summarise("relative difference", [v["rel_diff"] for v in values]),
        _summarise("float32 ULP distance", [float(v["ulps"]) for v in values]),
        _summarise(
            "agreeing significant digits",
            [v["digits"] for v in values if math.isfinite(v["digits"])],
        ),
        "",
    ]

    worst = sorted(values, key=lambda v: v["rel_diff"], reverse=True)[:n_worst]
    lines.append(f"{len(worst)} largest relative differences")
    for v in worst:
        lines.append(
            f"  ch{v['detector_id']} {v['field']:18s} "
            f"A={v['a']!r:22s} B={v['b']!r:22s} "
            f"rel={v['rel_diff']:.3e}  {v['ulps']} float32 ulp"
        )
    lines.append("")

    if digits_a is not None and digits_b is not None:
        lines.append("significant digits as printed in the source JSON")
        for label, counts in ((f"A ({len(digits_a)} values)", digits_a),
                              (f"B ({len(digits_b)} values)", digits_b)):
            if counts:
                lines.append(
                    f"  {label}: min {min(counts)}, max {max(counts)}, "
                    f"{sorted(set(counts))}"
                )
            else:
                lines.append(f"  {label}: no values")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir_a", type=Path, default=DEFAULT_DIR_A, help="first baseline directory"
    )
    parser.add_argument(
        "--dir_b", type=Path, default=DEFAULT_DIR_B, help="second baseline directory"
    )
    parser.add_argument(
        "--pattern", default="*.json", help="glob picking the JSON files out"
    )
    parser.add_argument(
        "--n_worst",
        type=int,
        default=5,
        help="how many of the largest disagreements to list",
    )
    parser.add_argument(
        "--dump",
        type=Path,
        default=None,
        help="write the per-value comparison to this JSON file",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    files_a = baseline_files(args.dir_a, args.pattern)
    files_b = baseline_files(args.dir_b, args.pattern)
    print(f"A: {len(files_a)} files in {args.dir_a}")
    print(f"B: {len(files_b)} files in {args.dir_b}")

    baseline_a = load_baseline(files_a)
    baseline_b = load_baseline(files_b)

    report = compare_baselines(baseline_a, baseline_b)
    print()
    print(
        format_report(
            report,
            label_a=str(args.dir_a),
            label_b=str(args.dir_b),
            digits_a=printed_digits(files_a),
            digits_b=printed_digits(files_b),
            n_worst=args.n_worst,
        )
    )

    if args.dump is not None:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        args.dump.write_text(
            json.dumps({**report, "digits": None}, indent=2, default=str)
        )
        print(f"per-value comparison written to {args.dump}")


if __name__ == "__main__":
    main()
