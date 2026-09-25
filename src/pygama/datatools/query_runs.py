from __future__ import annotations

import os
import re
from collections.abc import Collection, Mapping
from concurrent.futures import Executor
from contextlib import ExitStack
from copy import copy
from pathlib import Path

import awkward as ak
import numpy as np
from dbetto import TextDB
from rich.console import Console
from rich.status import Status

from .cycle_record import CycleRecord
from .utils import _read_dataflow_config, _setup_executor, _setup_spinner, get_recursive, _tiers_to_dict


def query_runs(
    runs: str | None = None,
    *,
    dataflow_config: Path | str | Mapping = "$REFPROD/dataflow-config.yaml",
    group_by: str | Collection[str] | None = None,
    sort_by: str | Collection[str] = "cycle",
    cycle_def: str | None = None,
    tiers: str | Collection[str] | Mapping[str, str] | None = None,
    join: str = "inner",
    ignored_cycles: str | Collection[str] | None = None,
    processes: int | None = None,
    executor: Executor | None = None,
    library: str = "ak",
    progress: Status | Console | bool = True,
):
    """
    Query runs and return a table containing one entry for each cycle and data
    extracted from cycle names. Optionally apply a boolean selection of runs to
    include using an expression ``runs``.

    Run DB is built by recursively cycling through directories in one of the
    data tiers (using a list of excluded files from metadata). The fields are
    parsed from the hyphen-separated elements of cycle names (as defined by
    the `cycle-def` arg below).

    Parameters
    ----------
    runs
        boolean python expression for selecting runs, using column names defined
        in ``cycle_def`` as variables.

        Examples:

        - select calibration data from periods 6, 7 and 8 (assuming l200-style cycle names)::

            "period>='p06' and period<='p08' and datatype=='cal'"

        - select runs for detectors V01234A and V06789B from Th calibration data
          (using Hades data cycle name ``experiment-det-datatype-run-starttime``)::

            "det in ['V01234A', 'V06789B'] and datatype=='th_HS2_lat_psa'"

    dataflow_config
        config file of reference production. If not provided, use the environment
        variable ``$REFPROD`` as a directory, and find file ``dataflow-config.yaml``

    group_by
        if ``None`` (default) return a flat array with all cycles. If one or more fields
        are provided, group entries by these fields (using :meth:`ak.run_lengths`, so group
        consecutive equal values; this is done after sorting, so be careful if sorting
        changes order!) Fields that vary within groups will be un-flattened into 2-D ragged
        arrays. Note that ``runs`` query cannot act collectively on grouped cycles.

    sort_by
        field by which to sort table, or list of fields in order by priority

    cycle_def
        hyphen-separated names of fields in cycle names; names will be used for columns.
        By default get from dataflow-config.

        Examples:
        - ``experiment-period-run-datatype-cycle`` for a L200 cycle, e.g. ``l200-p03-r001-cal-19720101T000000Z``
        - ``experiment-chan-datatype-run-starttime`` for a Hades cycle, e.g. ``char_data-V05268A-th_HS2_lat_psa-r001-20201008T122118Z``

    tiers
        tiers used to find files. First tier in list is used to walk through
        directories to populate run DB. Remaining tiers are checked for presence of
        cycles; a cycle is only added if it exists for each tier. File relative path
        for each tier's file is added as a column called ``tier_[t]``. Can provide:
        - Mapping from tier name to path to root of tier
        - List of tier names/single tier name. Paths will be found in ``dataflow_config["paths"]``
        - ``None``: read from ``dataflow_config``; if ``tiers`` entry not found, use ``"raw"``

    join
        type of join to use for different tiers. Options:
        - ``inner`` (default): select only cycles that have files in all tiers
        - ``outer``: select all cycles with files in any tiers. Fill files not found with ``None``
        - ``[tier]``: name of tier to use for "left" join; fill files in other tiers not found with ``None``

    ignored_cycles
        path(s) in metadata to list(s) of ignored cycles. By default get from dataflow-config,
        or else do not skip any cycles.

    processes:
        number of processes. If ``None``, use number equal to threads available
        to ``executor`` (if provided), or else do not parallelize

    executor:
        :class:`concurrent.futures.Executor` object for managing parallelism.
        If ``None``, create a :class:`concurrent.futures.`ProcessPoolExecutor`
        with number of processes equal to ``processes``.

    library
        format of returned table. Can be ``ak`` (default), ``pd`` or ``np``

    progress:
        if ``True`` draw progress spinner; can also provide a :class:`rich.Status`
        or:class:`rich.Console`
    """
    with ExitStack() as stack:
        _, executor = _setup_executor(stack, processes, executor)
        progress = _setup_spinner(stack, progress)
        dataflow_config, df_paths, query_config = _read_dataflow_config(dataflow_config)

        if cycle_def is None:
            if "cycle_def" not in query_config:
                msg = "cycle_def must be provided either as kwarg or in dataflow_config"
                raise ValueError(msg)
            cycle_def = query_config["cycle_def"]

        tiers = _tiers_to_dict(tiers, df_paths, query_config)

        if ignored_cycles is None:
            ignored_cycles = query_config.get("ignored_cycles", None)

        if join in ("inner", "outer"):
            # fancy one-liner to split first item from remaining items
            this_tier, other_tiers = (next(it:=iter(tiers.items())), dict(it))
        elif this_tier := next((t for t in tiers.items() if join == t[0]), False):
            # if join is a tier name, find the matching entry in tiers
            other_tiers = {t:p for t, p in tiers.items() if t != this_tier[0]}
        else:
            msg = f"invalid join argument {join}. Valid options: inner, outer, {", ".join(tiers)}"
            raise ValueError(msg)
        base_path = this_tier[1]

        # Get list of removed cycles if it exists
        if ignored_cycles is not None:
            if isinstance(ignored_cycles, str):
                ignored_cycles = [ignored_cycles]
            meta = TextDB(df_paths["metadata"], lazy=True)
            removed = set()
            for iclist in ignored_cycles:
                removed |= set(get_recursive(meta, iclist))
        else:
            removed = set()

        col_names = cycle_def.split("-")
        records = []

        for dirpath, dirnames, files in os.walk(base_path, followlinks=True):
            relpath = os.path.relpath(
                dirpath, base_path
            )  # get rid of base_path and the following slash

            # Prune subdirectories that are not in all tiers
            if join == "inner":
                for subdir in copy(dirnames):
                    if not all(
                        Path(p, relpath, subdir).is_dir() for _, p in other_tiers.items()
                    ):
                        dirnames.remove(subdir)

            if len(files) == 0:
                continue

            if executor is None:
                records += _get_run_records_loop(
                    files,
                    relpath,
                    col_names,
                    this_tier,
                    other_tiers if join != "outer" else dict(),
                    join == "inner",
                    removed,
                    runs,
                )
            else:
                records.append(
                    executor.submit(
                        _get_run_records_loop,
                        files,
                        relpath,
                        col_names,
                        this_tier,
                        other_tiers if join != "outer" else dict(),
                        join == "inner",
                        removed,
                        runs,
                    )
                )

        if executor is not None:
            records = [r for recs in records for r in recs.result()]

        # If outer join, recursively query other tiers and perform outer join
        if join == "outer" and len(other_tiers) > 0:
            other = query_runs(
                runs=runs,
                dataflow_config=dataflow_config,
                group_by=None,
                sort_by=sort_by,
                tiers=other_tiers,
                join="outer",
                ignored_cycles=ignored_cycles,
                processes=processes,
                executor=executor,
                library="pd",
                progress=progress,
            )

            if len(other) == 0:
                records = [r | {f"tier_{t}": None for t in other_tiers} for r in records]
            elif len(records) == 0:
                records = [r | {f"tier_{this_tier}": None} for r in other.to_dict(orient="records")]
            else:
                records = ak.to_dataframe(records)
                records = records.merge(
                    other,
                    on=["cycle", "relpath", *col_names],
                    how="outer",
                )
                records.where(records.notna(), None, inplace=True)
                records = records.to_dict(orient="records")

        # Format and return results
        records.sort(
            key=lambda rec: (
                rec[sort_by]
                if isinstance(sort_by, str)
                else [rec[sb] for sb in sort_by]
            )
        )
        result = ak.Array(records)

        if group_by is not None:
            if isinstance(group_by, str):
                lengths = [np.cumsum(ak.run_lengths(result[group_by]))]
            else:
                lengths = [np.cumsum(ak.run_lengths(result[f])) for f in group_by]
            lengths = np.unique(np.concatenate([0, *lengths]))
            result = ak.unflatten(result, lengths[1:] - lengths[:-1])
            result = ak.Array(
                {
                    f: ak.firsts(result[f])
                    if ak.all(ak.all(result[f] == ak.firsts(result[f]), axis=1), axis=0)
                    else result[f]
                    for f in result.fields
                }
            )

        if library == "ak":
            return result
        if library == "pd":
            return ak.to_dataframe(result)
        if library == "np":
            return ak.to_numpy(result)
        msg = "library must be 'ak', 'pd' or 'np'"
        raise ValueError(msg)


def list_run_fields(
    dataflow_config: Path | str | Mapping = "$REFPROD/dataflow-config.yaml",
    cycle_def: str | None = None,
    tiers: str | Collection[str] | Mapping[str, str] | None = None,
) -> list[str]:
    """
    List the fields that are available to :meth:`query_runs`.

    Parameters
    ----------
    dataflow_config
        config file of reference production. If not provided, use the environment
        variable ``$REFPROD`` as a directory, and find file ``dataflow-config.yaml``

    cycle_def
        hyphen-separated names of fields in cycle names; names will be used for columns.
        By default get from dataflow-config.

        Examples:
        - ``experiment-period-run-datatype-cycle`` for a L200 cycle, e.g. ``l200-p03-r001-cal-19720101T000000Z``
        - ``experiment-chan-datatype-run-starttime`` for a Hades cycle, e.g. ``char_data-V05268A-th_HS2_lat_psa-r001-20201008T122118Z``

    tiers
        tiers used to find files. First tier in list is used to walk through
        directories to populate run DB. Remaining tiers are checked for presence of
        cycles; a cycle is only added if it exists for each tier. File relative path
        for each tier's file is added as a column called ``tier_[t]``. Can provide:
        - Mapping from tier name to path to root of tier
        - List of tier names/single tier name. Paths will be found in ``dataflow_config["paths"]``
        - ``None``: read from ``dataflow_config``; if ``tiers`` entry not found, use ``"raw"``
    """
    _, df_paths, query_config = _read_dataflow_config(dataflow_config)

    if cycle_def is None:
        if "cycle_def" not in query_config:
            msg = "cycle_def must be provided either as kwarg or in dataflow_config"
            raise ValueError(msg)
        cycle_def = query_config["cycle_def"]
    tiers = _tiers_to_dict(tiers, df_paths, query_config)

    return {"relpath", "cycle"} | set(cycle_def.split("-")) | tiers.keys()


def _get_run_records_loop(
    files: list[str],
    relpath: str,
    col_names: list[str],
    this_tier: dict[str, str],
    other_tiers: dict[str, str],
    inner_join: bool,
    removed: set[str],
    runs,
):
    # Worker for query_runs to build a list of records for a directory
    records = []

    for f in sorted(files):
        # check if file name matches expected cycle patterns
        match = CycleRecord.parse_cycle.search(f)
        if not match:
            continue
        cycle_name = match.group(1)
        if cycle_name in removed:
            continue

        # Create the record
        record = {
            "relpath": relpath,
            "cycle": cycle_name,
            f"tier_{this_tier[0]}": f"{this_tier[1]}/{relpath}/{f}",
        }
        try:
            CycleRecord.update_cycle_fields(record, col_names)
        except ValueError:
            continue

        # evaluate the selection
        select_run = eval(runs, {}, record) if runs else True
        if not bool(select_run):
            continue

        # update record with other tiers
        try:
            CycleRecord.update_tiers(record, other_tiers, raise_on_missing=inner_join)
        except FileNotFoundError:
            continue

        records.append(record)
    return records
