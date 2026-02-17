from __future__ import annotations

import logging
from collections import namedtuple

import numpy as np
import pandas as pd
from lgdo.types import Table, VectorOfVectors

# later on we might want a tcm class, or interface / inherit from an entry list
# class. For now we just need the key clustering functionality

log = logging.getLogger(__name__)

coin_groups = namedtuple("coin_groups", ["name", "window", "window_ref"])


def _make_structured_sort_key(df: pd.DataFrame, sort_cols: list[str]) -> np.ndarray:
    arrays = [df[c].to_numpy() for c in sort_cols]
    key = np.empty(
        len(df), dtype=[(c, arr.dtype) for c, arr in zip(sort_cols, arrays, strict=True)]
    )
    for c, arr in zip(sort_cols, arrays, strict=True):
        key[c] = arr
    return key


def _sort_tcm_chunk(df: pd.DataFrame, sort_cols: list[str]) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df

    # Fast-path: if primary key is already monotonic, only sort within equal-key runs
    primary = sort_cols[0]
    primary_arr = df[primary].to_numpy()
    if np.all(primary_arr[:-1] <= primary_arr[1:]):
        change = np.flatnonzero(primary_arr[1:] != primary_arr[:-1]) + 1
        starts = np.concatenate(([0], change))
        stops = np.concatenate((change, [len(df)]))

        order_parts: list[np.ndarray] = []
        if len(sort_cols) == 1:
            order = np.arange(len(df), dtype=np.int64)
        else:
            rest_cols = sort_cols[1:]
            rest_arrays = {c: df[c].to_numpy() for c in rest_cols}

            for s, e in zip(starts, stops, strict=True):
                if e - s <= 1:
                    order_parts.append(np.arange(s, e, dtype=np.int64))
                    continue

                keys = tuple(rest_arrays[c][s:e] for c in rest_cols[::-1])
                order_parts.append(np.lexsort(keys).astype(np.int64) + s)

            order = np.concatenate(order_parts) if order_parts else np.arange(len(df))

        return df.take(order).reset_index(drop=True)

    return df.sort_values(sort_cols, kind="mergesort", ignore_index=True)


def _merge_sorted_tcm(tcm: pd.DataFrame, new_tcm: pd.DataFrame, sort_cols: list[str]) -> pd.DataFrame:
    if tcm is None:
        return new_tcm
    if new_tcm is None:
        return tcm

    a_key = _make_structured_sort_key(tcm, sort_cols)
    b_key = _make_structured_sort_key(new_tcm, sort_cols)

    n = len(tcm)
    m = len(new_tcm)

    # Insert each b element into a; b is sorted so insertion positions are nondecreasing.
    pos_in_a = np.searchsorted(a_key, b_key, side="right")
    pos_b = pos_in_a + np.arange(m, dtype=np.int64)

    is_b = np.zeros(n + m, dtype=bool)
    is_b[pos_b] = True

    order = np.empty(n + m, dtype=np.int64)
    order[~is_b] = np.arange(n, dtype=np.int64)
    order[is_b] = n + np.arange(m, dtype=np.int64)

    return (
        pd.concat([tcm, new_tcm], ignore_index=True, copy=False)
        .take(order)
        .reset_index(drop=True)
    )


def generate_tcm_cols(
    iterators: list,
    coin_windows: float = 0,
    table_keys: list[int] = None,
    row_in_tables: list[int] = None,
    fields: list[str] = None,
) -> dict[np.ndarray]:
    r"""Generate the columns of a time coincidence map.

    Generate the columns of a time coincidence map from a list of arrays of
    coincidence data (e.g. hit times from different channels). Returns 3
    :class:`numpy.ndarray`\ s representing a vector-of-vector-like structure:
    two flattened arrays ``table_key`` (e.g. channel number) and  ``row_in_table``
    (e.g. hit index) that specify the location in the input ``coin_data`` of
    each datum belonging to a coincidence event, and a ``cumulative_length``
    array that specifies which rows of the other two output arrays correspond
    to which coincidence event. These can be used to retrieve other data at
    the same tier as the input data into coincidence structures.

    The 0'th entry of ``cumulative_length`` contains the number of hits in the
    zeroth coincidence event, and the i'th entry is set to
    ``cumulative_length[i-1]`` plus the number of hits in the i'th event.
    Thus, the hits of the i'th event can be found in rows
    ``cumulative_length[i-1]`` to ``cumulative_length[i] - 1`` of ``table_key``
    and ``row_in_table``.

    An example: ``cumulative_length = [4, 7, ...]``.  Then rows 0 to 3 in
    `table_key` and `row_in_table` correspond to the hits in event 0, rows 4 to 6
    correspond to event 1, and so on.

    Makes use of :func:`pandas.concat`, :meth:`pandas.DataFrame.sort_values`,
    and :meth:`pandas.DataFrame.diff` functions:

    - pull data into a :class:`pandas.DataFrame`
    - sort events by strictly ascending value of `coin_col`
    - group hits if the difference in `coin_data` is less than `coin_window`

    Parameters
    ----------
    coin_data
        a list of arrays of the data to be clustered.
    coin_window
        the clustering window width. `coin_data` within the `coin_window` get
        aggregated into the same coincidence cluster. A value of ``0`` means an
        equality test.
    window_ref
        when testing one datum for inclusion in a cluster, test if it is within
        `coin_window` of

        - ``"first"`` -- the first element in the cluster (rigid window width)
        - ``"last"`` -- the last element in the cluster (window grows until two
          data are separated by more than coin_window)

    table_keys
        if provided, use `table_keys` in place of "index in coin_data" as the
        integer corresponding to each element of `coin_data` (e.g. a channel
        number).
    row_in_tables
        if provided, use these values in places of the ``DataFrame`` index for
        the return values of `row_in_table`.

    Returns
    -------
    col_dict
        keys are ``cumulative_length``, ``table_key``, and  ``row_in_table``.
        ``cumulative_length`` specifies which rows of the other two output
        arrays correspond to which coincidence event. ``table_key`` and
        ``row_in_table`` specify the location in ``coin_data`` of each datum
        belonging to the coincidence event.
    """
    if isinstance(iterators, list):
        iterators = np.array(iterators)

    tcm = None
    at_end = np.zeros(len(iterators), dtype=bool)
    skip_mask = np.zeros(len(iterators), dtype=bool)
    buffer = None
    if table_keys is None:
        table_keys = np.arange(0, len(iterators))
    while not at_end.all():
        curr_mask = ~skip_mask & ~at_end
        dfs = []
        for _ii, it in enumerate(iterators[curr_mask]):
            ii = np.where(curr_mask)[0][_ii]
            try:
                buffer = it.__next__()
                buf_len = len(buffer)
                start = it.current_i_entry
            except StopIteration:
                at_end[ii] = True
                continue
            if buf_len < it.buffer_len:
                at_end[ii] = True
            table_key = table_keys[ii]
            table_key = np.full(buf_len, table_key, dtype=int)
            buffer = buffer.view_as("pd")[:buf_len]
            buffer["table_key"] = table_key
            if row_in_tables is not None:
                buffer["row_in_table"] = row_in_tables.astype(int)[ii][
                    start : start + buf_len
                ]
            else:
                buffer["row_in_table"] = np.arange(start, start + buf_len, dtype=int)
            if len(buffer) > 0:
                dfs.append(buffer)  # don't copy the data!

        if at_end.all() and len(dfs) == 0 and tcm is None:
            break

        sort_cols = [entry.name for entry in coin_windows] + ["table_key"]

        log.debug("sorting new dfs")
        if len(dfs) > 0:
            new_tcm = _sort_tcm_chunk(
                pd.concat(dfs, ignore_index=True, copy=False), sort_cols
            )
        else:
            new_tcm = None
        log.debug("sorting new dfs: done")

        log.debug("merging sorted data")
        tcm = _merge_sorted_tcm(tcm, new_tcm, sort_cols)
        log.debug("merging sorted data: done")

        # define mask, true when new event, false if part of same event
        mask = np.zeros(len(tcm) - 1, dtype=bool)
        for entry in coin_windows:
            diffs = np.diff(tcm[entry.name])
            if entry.window_ref == "last":
                mask = mask | (diffs > entry.window)
            else:
                raise NotImplementedError(f"window_ref {entry.window_ref}")

        # grab up to evt including last instance of a channel to know that all channels
        # have been included in previous evts
        last_instance = {
            arr_id: index for index, arr_id in enumerate(tcm.table_key.to_numpy())
        }
        log.debug(f"last instance: {last_instance}")

        for i, entry in enumerate(table_keys):
            if entry not in last_instance:
                last_instance[entry] = np.inf
            if at_end[i]:
                last_instance[entry] = np.inf

        if len(np.array(table_keys)[~at_end]) > 1:
            comp_chan = np.array(table_keys)[~at_end][0]
            skip_mask = np.array(
                [(last_instance[arr] >= last_instance[comp_chan]) for arr in table_keys]
            )
            if skip_mask.all():
                skip_mask = np.zeros(len(table_keys), dtype=bool)
        else:
            skip_mask = np.zeros(len(table_keys), dtype=bool)

        # want to write entries only up to last entry of a channel to ensure all included in evt
        if at_end.all():
            log.debug("at end, writing all entries")
            write_mask = mask
            last_entry = None
        else:
            last_instance = int(np.min([last_instance[arr] for arr in table_keys]))
            last_entry = np.where(mask[:last_instance])[0]
            log.debug(f"last instance: {last_instance}")
            log.debug(f"last entry: {last_entry}")

            if len(last_entry) == 0:
                log.debug("last entry 0, going to next iteration")
                log.debug(tcm)
                continue
            else:
                last_entry = last_entry[-1] + 1
                write_mask = mask[:last_entry]
                if len(write_mask) == 0:
                    log.debug("no entries, going to next iteration")
                    log.debug(tcm)
                    continue

        # get cumulative_length
        cumulative_length = np.array(np.where(write_mask)[0]) + 1
        if at_end.all():
            cumulative_length = np.append(cumulative_length, len(write_mask) + 1)

        out_tbl = Table(size=len(cumulative_length))
        out_tbl.add_field(
            "table_key",
            VectorOfVectors(
                cumulative_length=cumulative_length,
                flattened_data=tcm["table_key"].to_numpy()[:last_entry],
            ),
        )
        out_tbl.add_field(
            "row_in_table",
            VectorOfVectors(
                cumulative_length=cumulative_length,
                flattened_data=tcm["row_in_table"].to_numpy()[:last_entry],
            ),
        )
        if fields is not None:
            for f in fields:
                out_tbl.add_field(
                    f,
                    VectorOfVectors(
                        cumulative_length=cumulative_length,
                        flattened_data=tcm[f].to_numpy()[:last_entry],
                    ),
                )
        if last_entry is None:
            tcm = None
        else:
            tcm = tcm[last_entry:]
        yield out_tbl
