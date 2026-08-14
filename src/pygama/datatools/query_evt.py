from __future__ import annotations

from collections.abc import Collection, Mapping
from concurrent.futures import Executor
from contextlib import ExitStack
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
from lh5 import LH5Iterator
from rich.console import Console
from rich.status import Status

from . import query_runs
from .utils import (
    _read_dataflow_config,
    _setup_executor,
    _setup_spinner,
    parse_query_paths,
)


def query_evt(
    fields: Collection[str],
    runs: str | ak.Array | Mapping[str, np.ndarray] | pd.DataFrame | None,
    events: str,
    *,
    dataflow_config: Path | str | Mapping = "$REFPROD/dataflow-config.yaml",
    tiers: Collection[str] | None = None,
    tables: Mapping[str, str] | None = None,
    return_query_vals: bool = False,
    processes: Executor | int | None = None,
    executor: Executor | None = None,
    library: str | None = None,
    progress: Status | Console | bool = True,
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

    events
        expression used to select data events for each run/channel. Expression
        can access values from the event tier. Parameters with aliases can be
        accessed using their on-disk field name or their alias. Awkward is available
        using ``ak``.

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
        mapping of tiers to format strings to access tables. Format strings may reference
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

    progress:
        if ``True`` draw progress bar; can also provide a :class:`rich.Status`
        or:class:`rich.Console`

    kwargs
        see :meth:`query_runs`
    """

    field_info = [parse_query_paths(f, fullmatch=True) for f in fields]
    events_fields = parse_query_paths(events)
    all_paths = {path for _, _, path in field_info + events_fields}

    with ExitStack() as stack:
        processes, executor = _setup_executor(stack, processes, executor)
        status = _setup_spinner(stack, progress)
        df_config, df_paths, query_config = _read_dataflow_config(dataflow_config)

        # Query (or convert) run_records
        if runs is None or isinstance(runs, str):
            run_records = query_runs(
                runs,
                dataflow_config=df_config,
                progress=status,
                tiers=tiers,
                **kwargs,
            )
        else:
            run_records = ak.Array(runs)
        if len(run_records) == 0:
            msg = "no run records were found"
            raise ValueError(msg)

        if status:
            status.update("Building iterator...", spinner="betaWave")

        if tables is None:
            if "tables" not in query_config:
                msg = "tables not found in dataflow_config; either provide as kwarg or add to config"
                raise ValueError(msg)
            tables = query_config["evt_tables"]

        events_fields = parse_query_paths(events)

        if tiers is None:
            tiers = query_config.get("evt_tiers", [])

        lh5_it = None
        for tier in tiers:
            tier_dir = df_paths[f"tier_{tier}"]
            lh5_files = [
                [f"{relpath}/{cycle}-tier_{tier}.lh5"]
                for relpath, cycle in zip(
                    run_records["relpath"], run_records["cycle"], strict=True
                )
            ]
            groups = [[tables[tier]]] * len(lh5_files)

            new_it = LH5Iterator(
                lh5_files,
                groups,
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

        if status:
            status.update("Querying data...")

        return lh5_it.query(
            events,
            fields=fields if not return_query_vals else None,
            processes=processes,
            executor=executor,
            library=library,
            progress=status.console if status else None,
        )
