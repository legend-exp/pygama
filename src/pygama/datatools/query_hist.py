from __future__ import annotations

from collections.abc import Collection, Mapping
from concurrent.futures import Executor
from contextlib import ExitStack
from inspect import signature
from pathlib import Path

import awkward as ak
import hist
import pandas as pd
from rich.console import Console
from rich.status import Status

from . import build_iterator, query_meta, query_runs
from .utils import _setup_executor, _setup_spinner, parse_query_paths


def query_hist(
    axes: Collection[hist.axis] | Mapping[str, hist.axis],
    runs: str | ak.Array | pd.DataFrame,
    channels: str,
    entries: str,
    *,
    dataflow_config: Path | str | Mapping = "$REFPROD/dataflow-config.yaml",
    processes: int | None = None,
    executor: Executor | None = None,
    progress: Status | Console | bool = True,
    **kwargs,
):
    """
    Query data from multiple tiers and metadata. Return a :class:`Hist`
    filled with data from hits from selected runs, channels, and data
    cuts. Values from metadata and parameters databases are accessed
    using (see :meth:query_meta)::

        [alias]@db_name.par_path

    In addition, parameters from data tables may be optionally aliased using::

        [alias]:nested.par_name

    If no alias is provided, then the on-disk name will be used; if a parameter
    is nested, then ``_`` will be used to separate levels (in the above example,
    the default alias would be ``nested_par_name``)

    Parameters
    ----------
    axes
        axis, list of axes, or mapping from data field to axis to use for histogram.
        If axis or list of axes, use ``axis.name`` for the field.  May include fields
        accessible with :meth:query_runs, :meth:query_meta, and fields in any data tier
        accessible by this method. See above for aliasing rules; if axis has no label,
        alias will be used

        Examples:

        - Energy histogram (300 bins ranging from 0 to 3000)::

            axis.Regular(300, 0, 3000, name="cuspEmax_ctc_cal", label="Energy (keV)")

        - 2-D histogram with energy on x-axis, and detector name on y-axis::

            {
                "cuspEmax_ctc_cal": axis.Regular(300, 0, 3000, label="Energy (keV)"),
                "@chan.name": axis.StrCategory(label="Detector", growth=True)"
            }

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

            "@chan.system=='geds' and @chan.type=='icpc' and @chan.analysis.usability=='on'"``

        - selects SiPM channel 10 and will only include runs where it is can be processed::

            "@chan.name=='S010' and @chan.analysis.processible"

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

    processes:
        number of processes. If ``None``, use number equal to threads available
        to ``executor`` (if provided), or else do not parallelize

    executor:
        :class:`concurrent.futures.Executor` object for managing parallelism.
        If ``None``, create a :class:`concurrent.futures.`ProcessPoolExecutor`
        with number of processes equal to ``processes``.

    progress:
        if ``True`` draw progress bar; can also provide a :class:`rich.Status`
        or:class:`rich.Console`

    kwargs
        see :meth:build_iterator, :meth:query_meta, :meth:query_runs, and :meth:Hist
    """

    if isinstance(axes, Mapping):
        field_info = [parse_query_paths(f, fullmatch=True) for f in axes]
        ax = list(axes.values())
    elif isinstance(axes, Collection):
        field_info = [parse_query_paths(a.name, fullmatch=True) for a in axes]
        ax = axes
    elif isinstance(axes, hist.axis.AxesMixin):
        field_info = [parse_query_paths(axes.name, fullmatch=True)]
        ax = [axes]
    else:
        msg = "axes must be axis or collection of axes"
        raise ValueError(msg)

    entries_fields = parse_query_paths(entries)

    with ExitStack() as stack:
        processes, executor = _setup_executor(stack, processes, executor)
        status = _setup_spinner(stack, progress)

        # split kwargs up based on which function they should feed into
        bi_kwargs = {}
        hist_kwargs = {}
        for k, kwarg in kwargs.items():
            if (
                k in signature(build_iterator).parameters
                or k in signature(query_meta).parameters
                or k in signature(query_runs).parameters
            ):
                bi_kwargs[k] = kwarg
            else:
                hist_kwargs[k] = kwarg

        lh5_it, alias_map = build_iterator(
            {f for f, _, _ in field_info + entries_fields},
            runs,
            channels,
            dataflow_config=dataflow_config,
            return_alias_map=True,
            processes=processes,
            executor=executor,
            progress=status,
            **bi_kwargs,
        )

        for (_, alias, path), a in zip(field_info, ax, strict=False):
            # add lh5 fields to alias map
            if path not in alias_map:
                alias_map[path] = alias if alias is not None else path
            if not a.label:
                a.label = alias_map.get(path)

        if status:
            status.update("Filling histogram...", spinner="betaWave")

        return lh5_it.hist(
            ax,
            where=entries,
            keys=[alias_map[path] for _, _, path in field_info],
            processes=processes,
            executor=executor,
            progress=status.console if status else None,
            **hist_kwargs,
        )
