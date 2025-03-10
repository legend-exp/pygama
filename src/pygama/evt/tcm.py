from __future__ import annotations

from collections import namedtuple

import numpy as np
import pandas as pd
from lgdo.types import Table, VectorOfVectors

# later on we might want a tcm class, or interface / inherit from an entry list
# class. For now we just need the key clustering functionality


coin_groups = namedtuple("coin_groups", ["name", "window", "ref"])


def generate_tcm_cols(
    iterators: list,
    coin_windows: float = 0,
    array_ids: list[int] = None,
    array_idxs: list[int] = None,
    fields: list[str] = None,
) -> dict[np.ndarray]:
    r"""Generate the columns of a time coincidence map.

    Generate the columns of a time coincidence map from a list of arrays of
    coincidence data (e.g. hit times from different channels). Returns 3
    :class:`numpy.ndarray`\ s representing a vector-of-vector-like structure:
    two flattened arrays ``array_id`` (e.g. channel number) and  ``array_idx``
    (e.g. hit index) that specify the location in the input ``coin_data`` of
    each datum belonging to a coincidence event, and a ``cumulative_length``
    array that specifies which rows of the other two output arrays correspond
    to which coincidence event. These can be used to retrieve other data at
    the same tier as the input data into coincidence structures.

    The 0'th entry of ``cumulative_length`` contains the number of hits in the
    zeroth coincidence event, and the i'th entry is set to
    ``cumulative_length[i-1]`` plus the number of hits in the i'th event.
    Thus, the hits of the i'th event can be found in rows
    ``cumulative_length[i-1]`` to ``cumulative_length[i] - 1`` of ``array_id``
    and ``array_idx``.

    An example: ``cumulative_length = [4, 7, ...]``.  Then rows 0 to 3 in
    `array_id` and `array_idx` correspond to the hits in event 0, rows 4 to 6
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

    array_ids
        if provided, use `array_ids` in place of "index in coin_data" as the
        integer corresponding to each element of `coin_data` (e.g. a channel
        number).
    array_idxs
        if provided, use these values in places of the ``DataFrame`` index for
        the return values of `array_idx`.

    Returns
    -------
    col_dict
        keys are ``cumulative_length``, ``array_id``, and  ``array_idx``.
        ``cumulative_length`` specifies which rows of the other two output
        arrays correspond to which coincidence event. ``array_id`` and
        ``array_idx`` specify the location in ``coin_data`` of each datum
        belonging to the coincidence event.
    """
    if isinstance(iterators, list):
        iterators = np.array(iterators)

    tcm = None
    at_end = np.zeros(len(iterators), dtype=bool)
    skip_mask = np.zeros(len(iterators), dtype=bool)

    if array_ids is None:
        array_ids = np.arange(0, len(iterators))

    while not at_end.any():
        curr_mask = ~skip_mask & ~at_end
        dfs = []
        for _ii, it in enumerate(iterators[curr_mask]):
            ii = np.where(curr_mask)[0][_ii]
            try:
                buffer, start, buf_len = it.__next__()
            except StopIteration:
                at_end[ii] = True
                continue
            if buf_len < it.buffer_len:
                at_end[ii] = True
            array_id = array_ids[ii]
            array_id = np.full(buf_len, array_id, dtype=int)
            buffer = buffer.view_as("pd")[:buf_len]
            buffer["array_id"] = array_id
            if array_idxs is not None:
                buffer["array_idx"] = array_idxs.astype(int)[ii][
                    start : start + buf_len
                ]
            else:
                buffer["array_idx"] = np.arange(start, start + buf_len, dtype=int)
            dfs.append(buffer)  # don't copy the data!

        if tcm is None:
            tcm = pd.concat(dfs).sort_values(
                [entry.name for entry in coin_windows] + ["array_id"]
            )
        else:
            tcm = pd.concat([tcm] + dfs).sort_values(
                [entry.name for entry in coin_windows] + ["array_id"]
            )

        mask = np.zeros(len(tcm) - 1, dtype=bool)
        for entry in coin_windows:
            diffs = np.diff(tcm[entry.name])
            if entry.window_ref == "last":
                mask = mask | (diffs > entry.window)
            else:
                raise NotImplementedError(f"window_ref {entry.window_ref}")
        # grab up to last instance of a channel to know that all channels have been included in evts
        last_instance = {
            arr_id: index for index, arr_id in enumerate(tcm.array_id.to_numpy())
        }

        for entry in array_ids:
            if entry not in last_instance:
                last_instance[entry] = np.inf

        # in next iteration read any channels < first otherwise read all
        skip_mask = np.array(
            [(last_instance[arr] > last_instance[array_ids[0]]) for arr in array_ids]
        )

        # want to write entries only up to last entry of a channel to ensure all included in evt
        if at_end.all():
            write_mask = mask
            last_entry = None
        else:
            last_instance = np.nanmin([int(last_instance[arr]) for arr in array_ids])
            last_entry = np.where(mask[last_instance:])[0]
            if len(last_entry) == 0:
                last_entry = None
                write_mask = mask
            else:
                last_entry = last_instance + last_entry[0]
                write_mask = mask[:last_entry]

        # get cumulative_length
        cumulative_length = np.array(np.where(write_mask)[0]) + 1
        cumulative_length = np.append(cumulative_length, len(write_mask) + 1)

        out_tbl = Table(size=len(cumulative_length))
        out_tbl.add_field(
            "array_id",
            VectorOfVectors(
                cumulative_length=cumulative_length,
                flattened_data=tcm["array_id"].to_numpy()[:last_entry],
            ),
        )
        out_tbl.add_field(
            "array_idx",
            VectorOfVectors(
                cumulative_length=cumulative_length,
                flattened_data=tcm["array_idx"].to_numpy()[:last_entry],
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
