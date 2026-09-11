from __future__ import annotations

import logging
import re
from copy import deepcopy
from typing import Literal

import awkward as ak
import lgdo
import lh5
import numpy as np
from lgdo.types import Struct, Table, VectorOfVectors

from . import tcm as ptcm

log = logging.getLogger(__name__)


def readd_attrs(final_table, input_table):
    if hasattr(input_table, "attrs"):
        final_table.attrs = input_table.attrs
    for key in input_table:
        if hasattr(input_table[key], "attrs"):
            final_table[key].attrs = input_table[key].attrs
        if isinstance(final_table[key], Table | Struct):
            readd_attrs(final_table[key], input_table[key])
        if isinstance(input_table[key], VectorOfVectors):
            final_table[key].flattened_data.attrs = input_table[
                key
            ].flattened_data.attrs
            final_table[key].cumulative_length.attrs = input_table[
                key
            ].cumulative_length.attrs


def _concat_tables(tbls):
    # one concatenate over the whole list, not a pairwise fold: the latter
    # recopies the accumulated result for every chunk, which is quadratic in
    # the chunk count and doubles peak memory
    out_tbl = ak.concatenate([tbl.view_as("ak") for tbl in tbls], axis=0)
    out_tbl = Table(col_dict=out_tbl)
    readd_attrs(out_tbl, tbls[0])
    return out_tbl


def build_tcm(
    input_tables: list[tuple[str, str | list[str]]],
    coin_cols: str | list[str],
    hash_func: str | None = r"\d+",
    coin_windows: float | list[float] = 0,
    window_refs: str | list[str] = "last",
    out_file: str | None = None,
    out_name: str = "tcm",
    channel_views: Literal["sparse", "dense", "all"] | None = None,
    view_group: str = "ch{key}",
    wo_mode: str = "write_safe",
    buffer_len: int | None = None,
    out_fields: str | list[str] | None = None,
) -> lgdo.Table | tuple[lgdo.Table, dict[str, np.ndarray]] | None:
    r"""Build a Time Coincidence Map (TCM).

    Given a list of input tables, create an output table containing an entry
    list of coincidences among the inputs. Uses
    :func:`.evt.tcm.generate_tcm_cols`. For use with the
    :class:`~.flow.data_loader.DataLoader`.

    Parameters
    ----------
    input_tables
        Each entry is ``(filename, table_name_pattern)``. ``table_name_pattern``
        may be a string or list of strings. All tables matching each pattern in
        ``filename`` will be used as input tables.
    coin_cols
        Name of the column (or columns) in each table used to build
        coincidences. All input tables must contain these columns.
    hash_func
        mapping of table names to integers for use in the TCM.  `hash_func` is
        a regexp pattern that acts on each table name. The default `hash_func`
        ``r"\d+"`` pulls the first integer out of the table name. Setting to
        ``None`` will use a table's index in `input_tables`.
    coin_windows
        Width of the clustering window(s). If a single value is supplied it will
        be used for all ``coin_cols``.
    window_refs
        Window reference for the clustering window. Currently only ``"last"`` is
        implemented.
    out_file
        name (including path) for the output file. If ``None``, no file will be
        written; the TCM will just be returned in memory.
    out_name
        name for the TCM table in the output file.
    channel_views
        mode for creating View of TCM for each channel, with only events
        containing that channel. Options:
        - ``"sparse"``: use in sparse trigger mode; a minority of events contain each channel
        - ``"dense"``: use in global trigger mode; (almost) all events contain (almost) all channels
        - ``"all"``: use in global trigger mode; assume all events are contained in all channels;
            if this is not the case, a RunTimeError will be raised!
        - ``"none"`` or ``None``: no channel views
    view_group
        format string for name of group containing view for each channel. View will
        be named ``out_name``. Can include the format specifiers:
        - ``key``: the index or hash integer of the channel
        - ``table``: the name of the input table
        Default: ``"ch{key}"`` (e.g. ``ch12345678/tcm``)
    wo_mode
        mode to send to :meth:`~lh5.io.store.LH5Store.write`.

    out_fields
        Optional additional fields to propagate from the input tables into the
        output TCM.

    Returns
    -------
    ``(lgdo.Table, dict[str, np.ndarray])`` or ``lgdo.Table`` or ``None``
        If we are outputting to a file, return ``None``. Otherwise,
        if ``out_file`` is ``None`` the resulting TCM is returned as a
        :class:`lgdo.Table`; if ``channel_views`` is additionally enabled,
        also return a ``dict`` from ``channel_name`` to entry list.

    See Also
    --------
    .tcm.generate_tcm_cols
    """
    # hash_func: later can add list or dict or a function(str) --> int.

    if not isinstance(coin_cols, list):
        coin_cols = [coin_cols]
    if not isinstance(coin_windows, list):
        coin_windows = [coin_windows]
    if not isinstance(window_refs, list):
        window_refs = [window_refs]
    if out_fields is not None and not isinstance(out_fields, list):
        out_fields = [out_fields]

    if len(coin_cols) != len(coin_windows):
        if len(coin_windows) == 1:
            coin_windows = coin_windows * len(coin_cols)
        else:
            msg = (
                "coin_cols and coin_windows must have the same length, "
                f"got {len(coin_cols)} and {len(coin_windows)}"
            )
            raise ValueError(msg)

    if len(coin_cols) != len(window_refs):
        if len(window_refs) == 1:
            window_refs = window_refs * len(coin_cols)
        else:
            msg = (
                "coin_cols and coin_windows must have the same length, "
                f"got {len(coin_cols)} and {len(window_refs)}"
            )
            raise ValueError(msg)

    iterators = []
    table_keys = []
    all_tables = []
    view_gps = {}

    # determine buffer length automatically
    if buffer_len is None:
        ntables = 0
        for filename, patterns in input_tables:
            patterns_list = [patterns] if isinstance(patterns, str) else patterns
            for pattern in patterns_list:
                ntables += len(lh5.ls(filename, lh5_group=pattern))

        n_fields = (
            2 + len(set(coin_cols + out_fields))
            if out_fields is not None
            else 2 + len(set(coin_cols))
        )
        buffer_len = int(10**7 / (ntables * n_fields))

    msg = f"buffer length is {buffer_len}"
    log.debug(msg)

    # loop over files
    for filename, patterns in input_tables:
        patterns_list = [patterns] if isinstance(patterns, str) else patterns

        # make a list of tables in the file
        tables_here = []
        for pattern in patterns_list:
            for table in lh5.ls(filename, lh5_group=pattern):
                tables_here.append(table)
                all_tables.append(table)

        msg = f"found tables {tables_here} in file {filename}"
        log.debug(msg)

        for table_idx, table in enumerate(tables_here):
            if hash_func is not None:
                if isinstance(hash_func, str):
                    table_key = int(re.search(hash_func, table).group())
                else:
                    msg = f"hash_func of type {type(hash_func).__name__}"
                    raise NotImplementedError(msg)
            else:
                table_key = table_idx

            msg = f"determined hash integer for {table}: {table_key}"
            log.debug(msg)

            h5py_open_mode = "a" if out_file == filename else "r"

            iterators.append(
                lh5.LH5Iterator(
                    filename,
                    table,
                    field_mask=coin_cols,
                    buffer_len=buffer_len,
                    h5py_open_mode=h5py_open_mode,
                )
            )
            table_keys.append(table_key)
            view_gps[view_group.format(key=table_key, table=table)] = ([], 0)

    coin_windows = [
        ptcm.coin_groups(n, w, r)
        for n, w, r in zip(coin_cols, coin_windows, window_refs, strict=False)
    ]

    tcm_gen = ptcm.generate_tcm_cols(
        iterators, coin_windows=coin_windows, table_keys=table_keys, fields=out_fields
    )

    tcm = None
    tcm_row = 0
    channel_views = channel_views.strip().lower() if channel_views is not None else None
    with lh5.LH5Store(keep_open=True, default_mode=wo_mode) as st:
        for out_tbl in tcm_gen:
            out_tbl.attrs.update(
                {"tables": str(all_tables), "hash_func": str(hash_func)}
            )

            # build entry list for each channel
            if channel_views not in ("none", None):
                for key, (view, (old_entries, offset)) in zip(
                    table_keys, view_gps.items(), strict=True
                ):
                    table_key = out_tbl.table_key.view_as("ak")
                    if channel_views == "sparse":
                        entries = np.flatnonzero(
                            np.array(ak.any(table_key == key, axis=-1))
                        )
                        entries += tcm_row
                        new_off = offset + len(old_entries)
                    elif channel_views == "dense":
                        # This builds a 2d array of ranges where our mask is True by identifying
                        # indices where mask changes between True and False
                        entries = np.reshape(
                            np.flatnonzero(
                                np.diff(
                                    np.concatenate(
                                        [
                                            [0],
                                            np.array(ak.any(table_key == key, axis=-1)),
                                            [0],
                                        ]
                                    )
                                )
                            ),
                            (-1, 2),
                        )
                        entries += tcm_row
                        new_off = offset + len(old_entries)
                    elif channel_views == "all":
                        # Build a single 2d array with all entries; overwrite for each iteration
                        if not ak.all(ak.any(table_key == key, axis=-1)):
                            msg = f"channel {key} not found in all events; channel_views='all' failed"
                            raise RuntimeError(msg)
                        entries = np.array([[0, len(table_key) + tcm_row]])
                        new_off = 0
                    else:
                        msg = f"unknown channel_views mode: {channel_views}"
                        raise ValueError(msg)
                    view_gps[view] = (entries, new_off)

            # Write to file
            if out_file is not None:
                st.write(
                    out_tbl,
                    out_name,
                    out_file,
                    write_start=tcm_row,
                )

                if channel_views not in ("none", None):
                    for view, (entries, offset) in view_gps.items():
                        st.write_view(
                            out_name,
                            entries,
                            out_name,
                            out_file,
                            group=view,
                            link_type="hard",
                            write_start=offset,
                            wo_mode=None if channel_views != "all" else "o",
                        )

            # build up the table in memory
            elif tcm is None:
                tcm = deepcopy(out_tbl)
                channel_entries = {k: v[0] for k, v in view_gps.items()}
            else:
                tcm.append(out_tbl)
                if channel_views not in ("none", None):
                    for ch_name, (new_entries, _) in view_gps.items():
                        if channel_views == "all":
                            channel_entries[ch_name] = new_entries
                        else:
                            channel_entries[ch_name] = np.append(
                                channel_entries[ch_name], new_entries, axis=0
                            )

            tcm_row += len(out_tbl)

    if tcm is None:
        return None
    if channel_views in ("none", None):
        return tcm
    return tcm, channel_entries
