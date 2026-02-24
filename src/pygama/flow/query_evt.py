import os
from collections.abc import Collection, Mapping
from concurrent.futures import Executor, ProcessPoolExecutor
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
from dbetto import Props
from legendmeta.query import _format_vars, parse_query_paths, query_runs
from lh5 import LH5Iterator


def query_evt(
    fields: Collection[str],
    runs: str | ak.Array | Mapping[np.ndarray] | pd.DataFrame | None,
    events: str,
    *,
    dataflow_config: Path | str | Mapping = "$REFPROD/dataflow-config.yaml",
    tiers: Collection[str] = None,
    tables: Collection[str] = None,
    return_query_vals: bool = False,
    processes: Executor | int = None,
    executor: Executor = None,
    library: str = None,
    **kwargs,
):
    """
    Query evt tier data. Return a table containing one entry for each event
    corresponding to the selected runs and data cuts, with columns
    for the requested data fields. Selections may be based on
    data fields in the evt tier or in the run descriptions.
    Values will be returned in a tabular format denoted by ``library``
    (default ``awkward.Array``). Parameters may be optionally aliased using:

        [alias]:nested.par_name

    If no alias is provided, then the on-disk name will be used.

    Parameters
    ----------
    fields
        list of fields to include in the table. May include fields accessible with
        :meth:query_runs, :meth:query_meta, and fields in any data tier accessible
        by this method. See above for aliasing rules.

    runs
        python boolean expression for selecting runs, using column names defined
        in ``cycle_def`` as variables. See :meth:query_runs

        Examples:

        - select calibration data from periods 6, 7 and 8 (assuming default cycle names)::

            "period>='p06' and period<='p08' and datatype=='cal'"

        - select runs for detectors V01234A and V06789B from Th calibration data
          (using Hades data cycle name ``experiment-det-datatype-run-starttime``)::

            "det in ["V01234A", "V06789B"] and datatype=='th_HS2_lat_psa'``

    entries
        expression used to select data entries for each run/channel. Expression
        can access values from any data tier, from all databases, and from
        run table. Parameters with aliases can be accessed using their on-disk
        field name or their alias.

        Examples:

        - select events with >100 keV of energy, with various event-level cuts applied::

            "(energy > 100) & (~coincident.puls) & (~coincident.spms) & (geds.multiplicity==1) & ak.all(geds.quality.is_bb_like, axis=-1)"

        - select hits with >500 keV of energy and manually applies the low A/E cut::

            ``"(cuspEmax_ctc_cal > 500) & (AoE_classifyer > @pars.pars.operations.AoE_Low_Cut.parameters.a)"``

    dataflow_config
        config file of reference production. If not provided, use the environment
        variable ``$REFPROD`` as a directory, and find file ``dataflow-config.yaml``

    tiers
        tiers to include

    tables
        list of format strings to access tables for each tier. Format strings may reference
        values from run or channel DBs. If no channel-wise information is included in the string
        the same table will be accessed for each channel (may be useful for evt tier). If ``None``,
        read from ``dataflow_config``. This is required.

    return_query_vals
        if ``True``, return values found in query as columns; else only return those in ``fields``

    processes:
        number of processes. If ``None``, use number equal to threads available
        to ``executor`` (if provided), or else do not parallelize

    executor:
        :class:`concurrent.futures.Executor` object for managing parallelism.
        If ``None``, create a :class:`concurrent.futures.`ProcessPoolExecutor`
        with number of processes equal to ``processes``.

    library
        format of returned table. Can be ``ak`` (default), ``pd`` or ``np``

    kwargs
        see :meth:`query_runs`
    """
    if isinstance(dataflow_config, (Path, str)):
        df_config = Props.read_from(
            os.path.expandvars(dataflow_config), subst_pathvar=True
        )
    elif isinstance(dataflow_config, Mapping):
        df_config = dataflow_config
    else:
        msg = "dataflow_config must be a str, Path, or Mapping"
        raise ValueError(msg)
    df_paths = df_config["paths"]
    query_config = df_config.get("query", {})

    field_info = [parse_query_paths(f, fullmatch=True) for f in fields]
    events_fields = parse_query_paths(events)
    all_paths = {path for _, _, path in field_info + events_fields}

    if processes is None and isinstance(executor, Executor):
        processes = executor._max_workers

    if executor is None and isinstance(processes, int):
        executor = ProcessPoolExecutor(processes)

    # Query (or convert) run_records
    if runs is None or isinstance(runs, str):
        run_records = query_runs(
            runs,
            dataflow_config=df_config,
            **kwargs,
        )
    else:
        run_records = ak.Array(runs)
    if len(run_records) == 0:
        msg = "no run records were found"
        raise ValueError(msg)

    if tiers is None:
        tiers = query_config.get("tiers", [])

    if tables is None:
        if "tables" not in query_config:
            msg = "tables not found in dataflow_config; either provide as kwarg or add to config"
            raise ValueError(msg)
        tables = query_config["tables"]

    events_fields = parse_query_paths(events)

    lh5_it = None
    for tier_key, tier_dir in df_paths.items():
        if not tier_key[:5] == "tier_":
            continue
        tier = tier_key[5:]
        if tiers and tier not in tiers:
            continue

        # keep only tiers with no channel information
        tab_name = tables[tier]
        if len(_format_vars(tab_name)) > 0:
            continue

        # broadcast groups and run_records
        lh5_files, groups, run_records = ak.broadcast_arrays(
            [
                [f"{relpath}/{cycle}-tier_{tier}.lh5"]
                for relpath, cycle in zip(run_records["relpath"], run_records["cycle"])
            ],
            [tab_name],
            run_records,
        )

        new_it = LH5Iterator(
            ak.to_list(lh5_files),
            ak.to_list(groups),
            base_path=tier_dir,
            group_data=run_records if lh5_it is None else None,
        )

        # only include if files exist and are required for some fields
        new_it.reset_field_mask(all_paths, warn_missing=False)
        if len(new_it.lh5_files) > 0 and len(new_it.field_mask) > 0:
            if lh5_it is None:
                lh5_it = new_it
            else:
                lh5_it.add_friend(new_it)

    lh5_it.reset_field_mask(all_paths, warn_missing=True)

    fields = {path: alias for _, alias, path in field_info}

    return lh5_it.query(
        events,
        fields=fields if not return_query_vals else None,
        processes=processes,
        executor=executor,
        library=library,
    )
