from __future__ import annotations

import logging
import sys
from collections.abc import Collection, Mapping
from concurrent.futures import Executor
from contextlib import ExitStack
from copy import copy

if sys.version_info >= (3, 12):
    from itertools import batched, repeat
else:
    from itertools import repeat
from pathlib import Path

import awkward as ak
import legendmeta
import numpy as np
import pandas as pd
from dbetto import TextDB
from legendmeta import MetadataRepository
from rich.console import Console
from rich.status import Status

from .query_runs import query_runs
from .utils import (
    _read_dataflow_config,
    _setup_executor,
    _setup_spinner,
    format_vars,
    get_recursive,
    parse_query_paths,
)

log = logging.getLogger(__name__)


def query_meta(
    fields: Collection[str],
    runs: str | ak.Array | pd.DataFrame,
    channels: str,
    *,
    dataflow_config: Path | str | Mapping = "$REFPROD/dataflow-config.yaml",
    group_chans: bool = False,
    tiers: Collection[str] | None = None,
    metadata: str | type | MetadataRepository = None,
    chan_db: str | None = None,
    meta_dbs: Mapping | None = None,
    return_query_vals: bool = False,
    return_alias_map: bool = False,
    processes: int | None = None,
    executor: Executor | None = None,
    library: str = "ak",
    progress: Status | Console | bool = True,
    **query_run_kwargs,
):
    """
    Query the metadata and pars data, returning a table containing one entry
    for each run/channel with the requested data fields. Can also provide
    boolean expression to select cycles based on data from runs table,
    and a boolean expression to select based on information about runs and
    channels found in metadata and parameter databases.

    Values from databases are referenced using:

        [alias]@db_name.par_path

    where:

    - ``alias``: optional alias to use as column name in returned table. If
      not provided, column name will be ``db_name_par_path``, replacing
      periods with underscores
    - ``@db_name``: name of data source. Data sources are found on disk using
      information in the dataflow config file (see ``dataflow_config``):

        - ``@chan``: channel database from ``metadata.channel_map()``
        - ``@par[_tier]``: parameter database from specified tier.
        - Additional data sources defined using the ``meta_dbs`` arg or
          ``meta_dbs`` entry in ``dataflow_config``

    - ``par_path``: path in database to par, using periods to separate fields

    Examples:

    - ``@chan.name``: name of channel; will be aliased to ``chan_name``
    - ``rid@chan.daq.raw_id``: DAQ id of channel; aliased to ``rid``
    - ``lt@run.livetime_in_s``: livetime from runinfo; aliased to ``lt``
    -  ``aoe_lo@par_hit.pars.operations.AoE_Low_Cut.parameters.a``: cut value
        for low A/E cut from hit tier

    Parameters
    ----------
    fields
        list of fields to include in the table. See above for description of
        syntax for naming data sources from metadata and parameter databases.

        Example::

            ["@chan.daq.rawid", "@run.livetime", "aoe_low_cut@par_hit.pars.operations.AoE_Low_Cut.parameters.a"]

    runs
        boolean python expression for selecting runs, using column names defined
        in ``cycle_def`` as variables. See :meth:`query_runs`

        Examples:

        - select calibration data from periods 6, 7 and 8 (assuming l200-style cycle names)::

            "period>='p06' and period<='p08' and datatype=='cal'"

        - select runs for detectors V01234A and V06789B from Th calibration data
          (using Hades data cycle name ``experiment-det-datatype-run-starttime``)::

            "det in ["V01234A", "V06789B"] and datatype=='th_HS2_lat_psa'"

    channels
        expression used to select channels for each run. Expression can
        access values from channel, metadata, and parameter databases (with
        the channel database ``@chan`` likely being the most useful)

        Examples:

        - select all ICPC channels for each run that are marked as usable::

            "@chan.type=='icpc' and @chan.analysis.usability=='on'"

        - select SiPM channel 10 and will only include runs where it is can be processed::

            "@chan.name=='S010' and @chan.analysis.processible"

        Note: if a parameter does not exist for a channel, it will evaluate to ``None``.
        If this causes an error to be thrown, this expression will evaluate to ``False``,
        excluding the channel. If an parameter always evaluates to None, it will raise
        an Exception.

    dataflow_config
        config file of reference production. If not provided, use the environment
        variable ``$REFPROD`` as a directory, and find file ``dataflow-config.yaml``

    tiers
        search only provided tiers for pars. If ``None`` search all found tiers.
        By default, get from dataflow-config.

        Examples: ``["raw", "dsp", "hit"]`` or ``["raw", "psp", "pht", "evt"]``

    metadata
        class or name of class to use to construct metadata

    chan_db
        format string for path in metadata to list of channels for a given cycle.
        Format string may reference values from the run DB. By default, get from
        dataflow-config or call :meth:`LegendMetadata.channelmap(starttime)`.

    meta_dbs
        mapping from database name (i.e. the thing after ``@``) to mapping of configuration
        parameters. Parameters are as follows:

            - path: path to root of database relative to root of metadata
            - cycle_entry [optional]: sub-path to entry for a given cycle. May be a format
              string with references to run DB fields. If not provided
              use :meth:`dbetto.TextDB.on(starttime)` to find the cycle entry
            - channel_entry [optional]: sub-sub-path to entry for a given channel. May be
              a format string with references to run DB and channel DB fields. If not
              provided and group_cycle is ``False``, use ``@chan.name``

        Examples::

            meta_dbs = {
                "runinfo": {
                    "path": "path/to/runinfo",
                    "cycle_entry": "{period}/{run}/{datatype}",
                },
                "chaninfo": {
                    "path": "path/to/chaninfo",
                    "cycle_entry": None, # use chaninfo.on(starttime)
                    "channel_entry": "@chan.name"
                }
                ...
            }

    group_chans
        if ``True``, return one entry for each run or group of runs, with channel data
        nested as ragged arrays. Else, return one entry for each channel/run.

    return_query_vals
        if ``True``, return values found in query as columns; else only return those in ``fields``

    return_alias_map
        if ``True``, return the pair ``(table, alias_map)`` where table is the
        normal output of this function and alias_map is a mapping from alias
        names to database paths

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

    query_run_kwargs
        see :meth:`query_runs`
    """
    with ExitStack() as stack:
        processes, executor = _setup_executor(stack, processes, executor)
        status = _setup_spinner(stack, progress)
        df_config, df_paths, query_config = _read_dataflow_config(dataflow_config)

        # Query (or convert) run_records
        if runs is None or isinstance(runs, str):
            run_records = query_runs(
                runs,
                dataflow_config=df_config,
                tiers=tiers,
                processes=processes,
                executor=executor,
                progress=status,
                **query_run_kwargs,
            )
        else:
            run_records = ak.Array(runs)
        if len(run_records) == 0:
            msg = (
                f'No run records were found for "{runs}". If your query seems correct, '
                "try setting tiers argument to omit tiers with no files. Call query_runs "
                'with join = "outer" to identify tiers with no files!'
            )
            raise ValueError(msg)

        # set up the status bar
        if status:
            status.update("Querying metadata...", spinner="betaWave")

        # setup metadata object
        if metadata is None:
            if "metadata" not in query_config:
                msg = "metadata must be provided either as kwarg or in dataflow_config"
                raise ValueError(msg)
            metadata = query_config["metadata"]
        if isinstance(metadata, str):
            meta = getattr(legendmeta, metadata)(df_paths["metadata"])
        elif isinstance(metadata, type):
            meta = metadata()
        elif isinstance(metadata, MetadataRepository):
            meta = metadata

        if not isinstance(meta, MetadataRepository):
            msg = "metadata must be a MetadataRepository derived class"
            raise ValueError(msg)

        # if using a string for chan_db, get it set up
        if chan_db is None:
            chan_db = query_config.get("chan_db", None)
        if chan_db is not None and not all(
            v in run_records.fields for v in format_vars(chan_db)
        ):
            msg = "chan_db must only reference values from run_db"
            raise ValueError(msg)

        # get list of fields needed and build mapping to column names
        col_name_map = {}
        col_list = set()
        chan_vars = parse_query_paths(channels)
        field_vars = [parse_query_paths(v, fullmatch=True) for v in fields]

        # capture alias@path.to.val into two variables
        for _, alias, path in chan_vars + field_vars:
            # map from path to alias
            if col_name_map.get(path) is None:
                # alias must be unique
                if alias is not None and any(
                    path != p and alias == a for p, a in col_name_map.items()
                ):
                    msg = f"alias {alias} already assigned"
                    raise ValueError(msg)
                col_name_map[path] = alias

            # path can only be aliased to a single name
            elif (
                path in col_name_map
                and alias is not None
                and alias != col_name_map[path]
            ):
                msg = f"{path} assigned multiple alias names ({alias}, {col_name_map[path]})"
                raise ValueError(msg)

        # Find all the un-aliased paths and assign them an alias
        for path, alias in col_name_map.items():
            if alias is None:
                new_alias = path.replace(".", "_").replace("@", "")
                col_name_map[path] = new_alias

        # add aliases to col_list
        for _, _, path in field_vars:
            col_list.add(col_name_map[path])

        for field, _, path in chan_vars:
            alias = col_name_map[path]
            channels = channels.replace(field, alias)
            if return_query_vals:
                col_list.add(alias)

        if return_query_vals:
            for f in run_records.fields:
                col_list.add(f)

        if meta_dbs is None:
            meta_dbs = query_config.get("meta_dbs", {})

        if tiers is None:
            tiers = query_config.get("tiers", [])

        # Build list of dbs to read from and perform checks on meta_var configuration
        db_list = {}
        for key, info in meta_dbs.items():
            if not any(v.split(".")[0] == f"@{key}" for v in col_name_map):
                continue

            path = info["path"]
            try:
                info["db"] = get_recursive(meta, path)
            except (KeyError, AttributeError) as e:
                msg = f"{path} not found in metadata"
                raise ValueError(msg) from e

            if info.get("cycle_entry"):
                if not all(
                    v in run_records.fields for v in format_vars(info["cycle_entry"])
                ):
                    msg = f"cycle_entry {info['cycle_entry']} for {key} references values not found in run DB"
                    raise ValueError(msg)
            elif "validity" not in info["db"]:
                msg = f"path {path} for {key} in metadata does not have validity file"
                raise ValueError(msg)

            if info.get("channel_entry") and not all(
                v in run_records.fields or v.startswith("@chan")
                for v in format_vars(info["channel_entry"])
            ):
                msg = f"channel_entry {info['channel_entry']} for {key} references values not found in run DB or channel DB"
                raise ValueError(msg)

            db_list[f"@{key}"] = info | {"db": get_recursive(meta, path)}

        # get the paths and groups corresponding to our query
        par_db_config = query_config.get("par_db", {"channel_entry": "{@chan.name}"})
        if "channel_entry" in par_db_config and not all(
            v in run_records.fields or v.startswith("@chan")
            for v in format_vars(par_db_config["channel_entry"])
        ):
            msg = f"channel_entry {par_db_config['channel_entry']} for par_db references values not found in run DB or channel DB"
            raise ValueError(msg)

        for t in tiers:
            if f"par_{t}" not in df_paths:
                continue

            if not any(v.split(".")[0] == f"@par_{t}" for v in col_name_map):
                db_list[f"@par_{t}"] = None
                continue

            try:
                db = TextDB(df_paths[f"par_{t}"], lazy=True)
                db_list[f"@par_{t}"] = par_db_config | {"db": db}
            except ValueError as e:
                msg = f"{df_paths[f'par_{t}']} not found for par_{t}"
                raise ValueError(msg) from e

        # Check that all parameters we try to read have a valid source
        for path in col_name_map:
            if path in run_records.fields:
                continue
            db_name = path.split(".")[0]
            if db_name != "@chan" and db_name not in db_list:
                msg = f"Could not find meta database for {path}. Available dbs:\n"
                msg += "@chan " + " ".join(db_list.keys())
                raise ValueError(msg)

        # Now run the query...
        if executor is None:
            records, eval_success, path_hits = _query_loop(
                run_records,
                col_list,
                channels,
                meta,
                chan_db,
                db_list,
                col_name_map,
                group_chans,
            )
        else:
            records = []
            eval_success = False
            path_hits = {}
            for rec, es, ph in executor.map(
                _query_loop,
                batched(run_records, int(np.ceil(len(run_records) / processes)))
                if sys.version_info >= (3, 12)
                else [
                    run_records[
                        i * int(np.ceil(len(run_records) / processes)) : (i + 1)
                        * int(np.ceil(len(run_records) / processes))
                    ]
                    for i in range(processes)
                ],
                repeat(col_list, processes),
                repeat(channels, processes),
                repeat(meta, processes),
                repeat(chan_db, processes),
                repeat(db_list, processes),
                repeat(col_name_map, processes),
                repeat(group_chans, processes),
            ):
                records += rec
                eval_success |= es
                for path, hits in ph.items():
                    path_hits[path] = path_hits.get(path, 0) + hits

        # if evaluating query was never successful...
        missing_params = [path for path, cts in path_hits.items() if cts == 0]
        if not eval_success or len(missing_params) > 0:
            msg = ""
            if not eval_success:
                msg = "Could not interpret channel query for any runs/channels:\n"
                msg += f"  {channels}\n"
            for path in missing_params:
                msg += f"{path} was not found for any run\n"
            raise ValueError(msg)

        # Format and return results
        if library == "ak":
            result = ak.Array(records)
        elif library == "pd":
            result = pd.json_normalize(records, sep="/")
            cols = result.columns.str.split("/", expand=True)
            # if columns are nested, they will be tuples; if nesting is uneven,
            # some tuples may have NaN; replace with '' to get better behavior
            if isinstance(cols[0], tuple):
                result.columns = pd.MultiIndex.from_tuples(
                    [[name if not pd.isna(name) else "" for name in c] for c in cols]
                )
        elif library == "np":
            if group_chans:
                msg = "library 'np' is not compatible with group_chans=True"
                raise ValueError(msg)

            # recursively walk through fields to produce nested dict of np arrays
            def ak_to_np(ak_tab):
                if len(ak_tab.fields) == 0:
                    return ak.to_numpy(ak_tab)
                return {f: ak_to_np(ak_tab[f]) for f in ak_tab.fields}

            result = ak_to_np(result)
        else:
            msg = "library must be 'ak', 'pd' or 'np'"
            raise ValueError(msg)

        if return_alias_map:
            return (result, col_name_map)
        return result


def _query_loop(
    run_records: Collection,
    col_list: set,
    channels: str,
    meta: MetadataRepository,
    chan_db: str | None,
    db_list: dict[str, TextDB],
    col_name_map: dict,
    group_chans: bool,
):
    # Now loop through the runs, perform channel queries, and fetch fields
    records = []
    path_hits = dict.fromkeys(
        col_name_map, 0
    )  # count number of times a value is found to give more helpful errors
    eval_success = False  # track if the eval ever succeeds
    cycle_dbs = {}
    chanlist = None
    for run_record in run_records:
        if isinstance(run_record, ak.Record):
            run_record = run_record.tolist()  # noqa: PLW2901
        time = run_record.get("starttime")
        while isinstance(time, list):
            time = time[0]

        if group_chans:
            record = copy(run_record)

        # Get pars DBs corresponding to current run
        for k, db_info in db_list.items():
            if db_info is None:
                continue
            try:
                if "cycle_entry" not in db_info:
                    if not time:
                        msg = f"starttime not found in rundb, cannot access {k} db"
                        raise ValueError(msg)
                    cycle_dbs[k] = db_info["db"].on(time)
                else:
                    cycle_dbs[k] = get_recursive(
                        db_info["db"], db_info["cycle_entry"].format(**run_record)
                    )
            except RuntimeError:
                # if there is no valid parameter database for this run...
                cycle_dbs[k] = None

        if not chan_db:
            if not time:
                msg = "starttime not found in rundb, cannot access channelmap"
                raise ValueError(msg)
            chanlist = meta.channelmap(on=time)
        else:
            chanlist = meta[chan_db.format(run_record)]

        # Get run DB entry corresponding to current run and get @run values
        for path, alias in col_name_map.items():
            # if it's in the run DB, it's already there
            if path in run_record:
                path_hits[path] += 1
                continue

            db_name = path.split(".")[0]
            if db_name == "@chan":
                continue
            db_info = db_list[db_name]

            if db_info.get("channel_entry"):
                continue

            try:
                run_record[alias] = eval(
                    path[1:], {}, {db_name[1:]: cycle_dbs[db_name]}
                )
                path_hits[path] += 1
            except (TypeError, KeyError, AttributeError):
                run_record[alias] = None

        ch_ct = 0
        for chan in chanlist.values():
            ch_record = copy(run_record)

            # Read values from database paths
            for path, alias in col_name_map.items():
                if path in run_record or alias in run_record:
                    continue

                db_name = path.split(".")[0]
                try:
                    if db_name == "@chan":
                        ch_db = chan
                    else:
                        ch_path = db_list[db_name]["channel_entry"].replace("@", "")
                        ch_db = get_recursive(
                            cycle_dbs[db_name], ch_path.format(chan=chan, **run_record)
                        )

                    ch_record[alias] = eval(path[1:], {}, {db_name[1:]: ch_db})
                    path_hits[path] += 1
                except (TypeError, KeyError, AttributeError):
                    ch_record[alias] = None

            # Evaluate the channel expression on the found values
            try:
                keep_record = bool(eval(channels, {}, ch_record))
            except (TypeError, NameError, KeyError, AttributeError):
                continue
            eval_success = True

            if keep_record:
                ch_ct += 1
                if group_chans:
                    for alias, param in ch_record.items():
                        if alias not in run_record:
                            record.setdefault(alias, []).append(param)
                else:
                    records.append(
                        {k: v for k, v in (ch_record).items() if k in col_list}
                    )

        if group_chans and ch_ct > 0:
            records.append({k: v for k, v in record.items() if k in col_list})

    return records, eval_success, path_hits
