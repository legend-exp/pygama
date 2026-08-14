from __future__ import annotations

from collections.abc import Collection, Mapping
from contextlib import ExitStack
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
from lh5 import LH5Iterator
from rich.console import Console
from rich.status import Status

from .query_meta import query_meta
from .query_runs import list_run_fields
from .utils import _read_dataflow_config, _setup_spinner, format_vars, parse_query_paths


def build_iterator(
    fields: Collection[str],
    runs: str | ak.Array | Mapping[str, np.ndarray] | pd.DataFrame,
    channels: str | ak.Array | Mapping[str, np.ndarray] | pd.DataFrame,
    *,
    dataflow_config: Path | str | Mapping = "$REFPROD/dataflow-config.yaml",
    tiers: Collection[str] | None = None,
    tables: Mapping[str, str] | None = None,
    return_alias_map: bool = False,
    progress: Status | Console | bool = True,
    **query_meta_kwargs,
):
    """
    build a :class:LH5Iterator to access data across multiple tiers and databases.

    Parameters
    ----------
    fields
        fields to include (across all tiers)

    runs
        python boolean expression for selecting runs, using column names defined
        in ``cycle_def`` as variables. See :meth:`query_runs`

        Examples:
            - ``"period>='p06' and period<='p08' and datatype=='cal'"`` selects calibration data from periods 6, 7 and 8 (assuming default cycle names)
            - ``"det in ['V01234A', 'V06789B'] and datatype=='th_HS2_lat_psa'"`` selects runs for detectors V01234A and V06789B from Th calibration data (using Hades data cycle name ``experiment-det-datatype-run-starttime``)

    channels
        expression used to select channels for each run. Expression can
        access values from all databases, as well as the run table.

        Examples:
            - ``"@chan.system=='geds' and @chan.type=='icpc' and @chan.analysis.usability=='on'"`` selects all ICPC detectors for each run that are marked as usable
            - ``"@chan.name=='S010' and @chan.analysis.processible"`` selects SiPM channel 10 and will only include runs where it is can be processed

        Note: if a parameter does not exist for a channel, it will evaluate to ``None``.
        If this causes an error to be thrown, this expression will evaluate to ``False``,
        excluding the channel. If an parameter always evaluates to False, it will raise
        an Exception.

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

    return_alias_map
        if ``True``, return the pair ``(table, alias_map)`` where table is the
        normal output of this function and alias_map is a mapping from alias
        names to database paths

    progress:
        if ``True`` draw progress spinner; can also provide a :class:`rich.Status`
        or:class:`rich.Console`

    query_meta_kwargs
        additional keyword arguments for :meth:`query_meta` and :meth:`query_runs`
    """
    with ExitStack() as stack:
        status = _setup_spinner(stack, progress)
        df_config, df_paths, query_config = _read_dataflow_config(dataflow_config)

        # identify fields we need to get from the metadata
        field_names = [parse_query_paths(f, fullmatch=True) for f in fields]
        run_fields = list_run_fields(
            dataflow_config=df_config,
            tiers=tiers,
            cycle_def=query_meta_kwargs.get("cycle_def"),
        )
        run_fields = {f for f, _, path in field_names if path in run_fields}
        meta_fields = {f for f, _, path in field_names if path[0] == "@"}

        if tables is None:
            if "tables" not in query_config:
                msg = "tables not found in dataflow_config; either provide as kwarg or add to config"
                raise ValueError(msg)
            tables = query_config["tables"]

        if tiers is None:
            tiers = query_config.get("tiers", [])

        gp_fields = {}
        for tier, tb in tables.items():
            fields = [parse_query_paths(f, fullmatch=True) for f in format_vars(tb)]
            meta_fields |= {f for f, _, path in fields if path[0] == "@"}
            gp_fields[tier] = fields

        run_data, alias_map = query_meta(
            meta_fields | run_fields | {"cycle", "relpath"},
            runs,
            channels,
            dataflow_config=df_config,
            tiers=tiers,
            group_chans=True,
            return_alias_map=True,
            progress=status,
            **query_meta_kwargs,
        )

        field_names = [alias_map.get(path, path) for _, _, path in field_names]

        if status:
            status.update("Building iterator...")

        lh5_it = None
        for tier in tiers:
            tier_dir = df_paths[f"tier_{tier}"]
            lh5_files = [
                [f"{relpath}/{cycle}-tier_{tier}.lh5"]
                for relpath, cycle in zip(
                    run_data["relpath"], run_data["cycle"], strict=True
                )
            ]
            tab_format = tables[tier]
            tab_fields = []

            for fname, _, path in gp_fields[tier]:
                tab_format = tab_format.replace(fname, alias_map[path])
                tab_fields.append(alias_map[path])
            if len(tab_fields) > 0:
                tab_vals = ak.Array(
                    dict(
                        zip(
                            tab_fields,
                            ak.broadcast_arrays(*[run_data[f] for f in tab_fields]),
                            strict=True,
                        )
                    )
                )
                groups = [
                    [
                        tab_format.format(**dict(zip(rec.fields, vals, strict=True)))
                        for vals in zip(*[rec[f] for f in rec.fields], strict=True)
                    ]
                    for rec in tab_vals
                ]
            else:
                lens = [1] * len(lh5_files)
                for f in run_data.fields:
                    if run_data[f].ndim >= 2:
                        lens = ak.num(run_data[f])
                        break
                groups = [[tab_format] * ln for ln in lens]

            new_it = LH5Iterator(
                lh5_files,
                groups,
                base_path=tier_dir,
                group_data=run_data if lh5_it is None else None,
            )

            # only include if files exist and are required for some fields
            new_it.reset_field_mask(field_names, warn_missing=False)
            if len(new_it.lh5_files) > 0 and len(new_it.field_mask) > 0:
                if lh5_it is None:
                    lh5_it = new_it
                else:
                    lh5_it.add_friend(new_it)

        if lh5_it is None:
            msg = "No LH5 files were found for the selected fields, runs, tiers, and channels."
            raise ValueError(msg)
        lh5_it.reset_field_mask(field_names, warn_missing=True)

        if return_alias_map:
            return lh5_it, alias_map
        return lh5_it
