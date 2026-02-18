from __future__ import annotations

import logging
from collections import namedtuple

import awkward as ak
import numpy as np
from lgdo.types import Table, VectorOfVectors

# later on we might want a tcm class, or interface / inherit from an entry list
# class. For now we just need the key clustering functionality

log = logging.getLogger(__name__)

coin_groups = namedtuple("coin_groups", ["name", "window", "window_ref"])


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

    This implementation uses Awkward Arrays for concatenation, sorting, and
    clustering.

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

    def _sort_tcm(arr: ak.Array, coin_windows_local) -> ak.Array:
        if arr is None or len(arr) == 0:
            return arr

        # primary keys are coin_cols in given order, with table_key as a tie-breaker
        table_key_np = ak.to_numpy(arr["table_key"])
        coin_key_nps = [ak.to_numpy(arr[entry.name]) for entry in coin_windows_local]
        order = np.lexsort([table_key_np] + list(reversed(coin_key_nps)))
        return arr[order]

    if isinstance(iterators, list):
        iterators = np.array(iterators)

    if coin_windows in (None, 0):
        coin_windows = []
    elif not isinstance(coin_windows, (list, tuple, np.ndarray)):
        coin_windows = [coin_windows]

    tcm = None
    at_end = np.zeros(len(iterators), dtype=bool)
    skip_mask = np.zeros(len(iterators), dtype=bool)
    buffer = None

    if table_keys is None:
        table_keys = np.arange(0, len(iterators))

    while not at_end.all():
        curr_mask = ~skip_mask & ~at_end
        arrays = []

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

            if buf_len <= 0:
                continue

            buf_ak = buffer.view_as("ak")[:buf_len]

            table_key = int(table_keys[ii])
            buf_ak = ak.with_field(
                buf_ak, np.full(buf_len, table_key, dtype=int), "table_key"
            )

            if row_in_tables is not None:
                row_in_table = row_in_tables.astype(int)[ii][start : start + buf_len]
            else:
                row_in_table = np.arange(start, start + buf_len, dtype=int)
            buf_ak = ak.with_field(buf_ak, row_in_table, "row_in_table")

            arrays.append(buf_ak)

        if at_end.all() and len(arrays) == 0 and tcm is None:
            break
        if len(arrays) == 0 and tcm is None:
            continue

        if tcm is None:
            tcm = ak.concatenate(arrays, axis=0) if len(arrays) > 0 else None
        else:
            if len(arrays) > 0:
                tcm = ak.concatenate([tcm] + arrays, axis=0)

        if tcm is None or len(tcm) == 0:
            continue

        tcm = _sort_tcm(tcm, coin_windows)

        # define mask, true when new event, false if part of same event
        mask = np.zeros(len(tcm) - 1, dtype=bool)
        for entry in coin_windows:
            diffs = np.diff(ak.to_numpy(tcm[entry.name]))
            if entry.window_ref == "last":
                mask = mask | (diffs > entry.window)
            else:
                raise NotImplementedError(f"window_ref {entry.window_ref}")

        # grab up to evt including last instance of a channel to know that all channels
        # have been included in previous evts
        table_key_np = ak.to_numpy(tcm["table_key"])
        last_instance = {arr_id: index for index, arr_id in enumerate(table_key_np)}
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
            last_instance_min = int(np.min([last_instance[arr] for arr in table_keys]))
            last_entry = np.where(mask[:last_instance_min])[0]
            log.debug(f"last instance: {last_instance_min}")
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
                flattened_data=ak.to_numpy(tcm["table_key"])[:last_entry],
            ),
        )
        out_tbl.add_field(
            "row_in_table",
            VectorOfVectors(
                cumulative_length=cumulative_length,
                flattened_data=ak.to_numpy(tcm["row_in_table"])[:last_entry],
            ),
        )
        if fields is not None:
            for f in fields:
                out_tbl.add_field(
                    f,
                    VectorOfVectors(
                        cumulative_length=cumulative_length,
                        flattened_data=ak.to_numpy(tcm[f])[:last_entry],
                    ),
                )

        if last_entry is None:
            tcm = None
        else:
            tcm = tcm[last_entry:]

        yield out_tbl
