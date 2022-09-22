from __future__ import annotations

import numpy as np
import pandas as pd

# later on we might want a tcm class, or interface / inherit from an entry list
# class. For now we just need the key clustering functionality


def generate_tcm_cols(
    coin_data: list[np.ndarray],
    coin_window: float = 0,
    window_ref: str = "last",
    array_ids: list[int] = None,
    array_idxs: list[int] = None,
) -> dict[np.ndarray]:
    r"""Generate the columns of a time coincidence map.

    Generate the columns of a time coincidence map from a list of arrays of
    coincidence data (e.g. hit times from different channels). Returns 3
    equal-length :class:`numpy.ndarray`\ s containing the coincidence index
    (e.g. event number), array ID (e.g. channel number), and array index (e.g.
    hit ID). These can be used to retrieve other data at the same tier as the
    input data into coincidence structures.

    Makes use of :func:`pandas.concat`, :meth:`pandas.DataFrame.sort_values`,
    :meth:`pandas.DataFrame.groupby`, and
    :meth:`pandas.DataFrame.cumsum`/:meth:`pandas.DataFrame.count` functions:

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
        keys are ``coin_idx``, ``array_id``, and  ``array_idx``. ``coin_idx``
        specifies which rows of the output arrays correspond to the which
        coincidence event ``array_id`` and ``array_idx`` specify the location
        in ``coin_data`` of each datum belonging to the coincidence event.
    """
    dfs = []
    for ii, array in enumerate(coin_data):
        array = np.array(array)
        array_id = array_ids[ii] if array_ids is not None else ii
        array_id = np.full_like(array, array_id)
        col_dict = {"array_id": array_id, "coin_data": array}
        if array_idxs is not None:
            col_dict["array_idx"] = array_idxs[ii]
        dfs.append(pd.DataFrame(col_dict, copy=False))

    # concat and sort
    tcm = pd.concat(dfs).sort_values(["coin_data", "array_id"])

    # compute coin_data diffs
    tcm["dcoin"] = tcm["coin_data"].diff()

    # window into coincidences
    # In the future, can add more options, like mean/median/mode
    if window_ref == "last":
        # create the event column by comparing the time since last event to the coincindence window
        tcm["coin_idx"] = (tcm.dcoin > coin_window).cumsum()
    else:
        raise NotImplementedError(f"window_ref {window_ref}")

    # now build the outputs
    cumulative_length = np.where(tcm.coin_idx.diff().to_numpy() != 0)[0]
    cumulative_length[:-1] = cumulative_length[1:]
    cumulative_length[-1] = len(tcm.coin_idx)
    array_id = tcm.array_id.to_numpy()
    array_idx = (
        tcm.array_idx.to_numpy() if "array_idx" in tcm else tcm.index.to_numpy()
    )  # beautiful!
    return {
        "cumulative_length": cumulative_length,
        "array_id": array_id,
        "array_idx": array_idx,
    }
