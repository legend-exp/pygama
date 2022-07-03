from __future__ import annotations

import numpy as np
import pandas as pd

# later on we might want a tcm class, or interface / inherit from an entry list
# class. For now we just need the key clustering functionality

def generate_tcm_cols(coin_data:list, coin_window:float=0, window_ref:str='last',
                      array_ids:list=None, array_idxs:list=None):
    """
    Generate the columns of a time coincidence map from a list of arrays of
    coincidence data (e.g. hit times from different channels). Returns 3
    equal-length ndarrays containing the coincidence index (e.g. event number),
    array id (e.g. channel number), and array index (e.g. hit id). These can be
    used to retrieve other data at the same tier as the input data into
    coincidence structures.

    Makes use of pandas DataFrame's concat, sort, groupby, and cumsum/count
    functions:
    - pull data into a DataFrame
    - Sort events by strictly ascending value of coin_col
    - Group hits if the difference in coin_data is less than coin_window

    Parameters
    ----------
    coin_data : list of ndarrays
        A list of ndarrays of the data to be clustered
    coin_window : float (optional)
        The clustering window width. coin_data within the coin_window get
        aggregated into the same coincidence cluster. A value of 0 means an
        equality test.
    window_ref : str
        When testing one datum for inclusion in a cluster, test if it is within
        coin_window of
        'first' -- the first element in the cluster (rigid window width) (not
        implemented yet)
        'last' -- the last element in the clustur (window grows until two data
        are separated by more than coin_window)
        In the future, can add more options, like mean/median/mode
    array_ids : list of ints or None
        If provided, use array_ids in place of "index in coin_data" as the
        integer corresponding to each element of coin_data (e.g. a channel
        number)
    array_idxs : list of indices or None
        If provided, use these values in places of df.index for the return
        values of array_idx

    Returns
    -------
    col_dict : dict of ndarrays
        keys are 'coin_idx', 'array_id', and 'array_idx'
        coin_idx specifies which rows of the output arrays correspond to the
        which coincidence event
        array_id and array_idx specify the location in coin_data of each datum
        belonging to the coincidence event
    """
    dfs = []
    for ii, array in enumerate(coin_data):
        array = np.array(array)
        array_id = array_ids[ii] if array_ids is not None else ii
        array_id = np.full_like(array, array_id)
        col_dict = {'array_id':array_id, 'coin_data':array}
        if array_idxs is not None: col_dict['array_idx'] = array_idxs[ii]
        dfs.append(pd.DataFrame(col_dict, copy=False))

    # concat and sort
    tcm = pd.concat(dfs).sort_values(['coin_data', 'array_id'])

    # compute coin_data diffs
    tcm['dcoin'] = tcm['coin_data'].diff()

    # window into coincidences
    if window_ref == 'last':
        # create the event column by comparing the time since last event to the coincindence window
        tcm['coin_idx'] = (tcm.dcoin > coin_window).cumsum()
    else:
        raise NotImplementedError(f'window_ref {window_ref}')

    # now build the outputs
    coin_idx = tcm.coin_idx.to_numpy()
    array_id = tcm.array_id.to_numpy()
    array_idx = tcm.array_idx.to_numpy() if 'array_idx' in tcm else tcm.index.to_numpy() # beautiful!
    return { 'coin_idx':coin_idx, 'array_id':array_id, 'array_idx':array_idx }
