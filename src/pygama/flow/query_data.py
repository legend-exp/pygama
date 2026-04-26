from collections.abc import Collection, Mapping
from concurrent.futures import Executor, ProcessPoolExecutor
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
from legendmeta.query import parse_query_paths

from . import build_iterator


def query_data(
    fields: Collection[str],
    runs: str | ak.Array | Mapping[np.ndarray] | pd.DataFrame,
    channels: str | ak.Array | Mapping[np.ndarray] | pd.DataFrame,
    entries: str,
    *,
    dataflow_config: Path | str | Mapping = "$REFPROD/dataflow-config.yaml",
    return_query_vals: bool = False,
    return_alias_map: bool = False,
    processes: Executor | int = None,
    executor: Executor = None,
    library: str = None,
    **kwargs,
):
    """
    Query data from multiple tiers and metadata. Return a table
    containing one entry for each hit corresponding to the
    selected runs, channels, and data cuts, with columns for
    the requested data fields. Selections may be based on
    data fields in any tier of data file, in select metadata
    tables, in the parameters databases, or in the run descriptions.
    Values will be returned in a tabular format denoted by ``library``
    (default ``awkward.Array``). Values from metadata and parameters
    databases are accessed using (see :meth:query_meta):

        [alias]@db_name.par_path

    In addition, parameters from data tables may be optionally aliased using:

        [alias]:nested.par_name

    If no alias is provided, then the on-disk name will be used; if a parameter
    is nested, then ``_`` will be used to separate levels (in the above example,
    the default alias would be ``nested_par_name``)

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

    channels
        expression used to select channels for each run. Expression can
        access values from all databases, as well as the run table.

        Examples:

        - select all ICPC detectors for each run that are marked as usable::

            "@det.system=='geds' and @det.type=='icpc' and @det.analysis.usability=='on'"``

        - selects SiPM channel 10 and will only include runs where it is can be processed::

            "@det.name=='S010' and @det.analysis.processible"

        Note: if a parameter does not exist for a channel, it will evaluate to ``None``.
        If this causes an error to be thrown, this expression will evaluate to ``False``,
        excluding the channel. If an parameter always evaluates to False, it will raise
        an Exception.

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

    kwargs
        see :meth:`build_iterator`, :meth:`query_meta` and :meth:`query_runs`
    """

    field_info = [parse_query_paths(f, fullmatch=True) for f in fields]
    entries_fields = parse_query_paths(entries)

    if processes is None and isinstance(executor, Executor):
        processes = executor._max_workers

    if executor is None and isinstance(processes, int):
        executor = ProcessPoolExecutor(processes)

    lh5_it, alias_map = build_iterator(
        {f for f, _, _ in field_info + entries_fields},
        runs,
        channels,
        dataflow_config=dataflow_config,
        return_alias_map=True,
        processes=processes,
        executor=executor,
        **kwargs,
    )

    fields = {}
    for _, alias, path in field_info:
        if path in alias_map:
            fields[alias_map[path]] = None
        else:
            fields[path] = alias

    ret = lh5_it.query(
        entries,
        fields=fields if not return_query_vals else None,
        processes=processes,
        executor=executor,
        library=library,
    )

    if return_alias_map:
        return ret, alias_map
    return ret
