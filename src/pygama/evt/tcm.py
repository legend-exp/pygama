def cluster_events(tb_list:list, ts_unit:float=1e-8, ch_col:str='channel',
                   ts_col:str='timestamp', coin_window:float=4e-6,
                   data_cols:list=None):
    """
    Create a time coincidence map (TCM), given a list of data tables from separate channels.
    Assume that all tables are from different channels in a SINGLE cycle file.
    Hopefully we won't need to extend this to event building across multiple cycles
    as in Majorana.
    - Sort events in DataFrames by strictly ascending timestamps
    - Group hits if the difference between times is less than `coin_window` (default length 4*us)
    - Don't allow events from the same channel into any event.
    - TODO: add a "t_offset" output column which is the time since the FIRST hit in the event.

    Parameters
    ----------
    tb_list : list
        A list of N tables containing 'channel' & 'timestamp' columns.
    ts_unit : float (optional)
        Conversion factor for timestamps in native units to SECONDS.
    ch_col : str (optional)
        Name of the common column to use as 'channel'.  Default is 'channel'.
    ts_col : str (optional)
        Name of the common column to use as 'timestamp'.  Default is 'timestamp'.
    coin_window : float (optional)
        The default clustering time in seconds. (4e-6 sec is good for HPGe detectors)
        If events in other channels occur within the first window, extend the window
        by this amount and search again.
    data_cols : list (optional)
        Copy over additional columns (such as DSP parameters) when we sort events together
        by timestamps.  This is handy for event building DSP files with pandas, then
        optionally converting back to LH5 tables for file i/o.

    Returns
    -------
    tcm : DataFrame
        Return a table with NEW columns: ['ix_evt','ix_hit','tcm_sec','tcm_dt','idx_row_{ch}'].
        If data_cols is set, these columns will be added to the input tables,
        usually dsp data.
    """
    if not isinstance(tb_list[0], pd.DataFrame):
        print("LH5 tables not supported yet, but easy, just need to convert them to DataFrame")
        return None

    # create 'ts_sec' timestamps for each channel, using the supplied conversion to seconds.
    for df in tb_list:
        df['tcm_sec'] = df[ts_col] * ts_unit

        # throw an error if we detect resetting timestamps
        ts = df['tcm_sec'].values
        tdiff = np.diff(ts)
        tdiff = np.insert(tdiff, 0 , 0)
        ix_resets = np.where(tdiff < 0)
        if len(ix_resets[0]) > 0:
            print('Warning! timestamps reset for this channel.  TC map will be total nonsense!  AAAAAHHH')

        # save the original index for reverse lookup
        chan = df[ch_col].unique()[0]
        df[f'ix_row_{chan}'] = df.index.values

    # # make a list of columns from the input tables we want to copy over
    # copy_cols = ['tcm_sec'] + ch_rows
    # if data_cols is not None:
    #     copy_cols.extend(data_cols)
    # copy_cols = sorted(list(set(copy_cols))) # drop duplicates

    # create a new dataframe where we SORT ALL ROWS by a strictly ascending timestamp
    dfs = tb_list
    tcm = pd.concat(dfs).sort_values('tcm_sec')
    tcm.reset_index(inplace=True, drop=True)

    # create the event column by comparing the time since last event to the coincindence window
    tcm['tcm_dt'] = tcm['tcm_sec'].diff()
    tcm['ix_evt'] = (tcm.tcm_dt > coin_window).cumsum()

    # create the sub-event column (groupbys are easy with pandas, hard with LH5 tables.)
    tcm['ix_hit'] = tcm.groupby(tcm.ix_evt).cumcount()

    return tcm
