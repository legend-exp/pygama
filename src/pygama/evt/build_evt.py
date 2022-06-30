#!/usr/bin/env python3
import argparse
import os
import time
from pprint import pprint

import h5py
import numpy as np
import pandas as pd

import pygama.lgdo.lh5_store as lh5
from pygama.evt.tcm import cluster_events


def build_evt(f_hit:str, f_evt:str=None, lh5_tables:list=None, copy_cols:list=None,
               overwrite:bool=True, evt_tb_name:str=None, builder_config:dict=None):
    """
    Given an input file containing separate Tables for each active data taker,
    create an output Table which merges all channels together, sorting everything
    by ascending timestamp, and grouping coincident hits in each channel together
    as "events".  The output table will contain columns ['ix_evt', 'ix_sub']
    denoting the event and sub-event (hit), in addition to any other columns
    we desire to copy over into the output file.

    Parameters
    ----------
    f_hit : str
        Input file containing multiple LH5 tables, one for each channel.
        Usually a RAW or DSP file.
    f_evt : str
        Output file name.
    lh5_tables : list (optional)
        Specify the list of tables to analyze, instead of detecting them
        automatically by default.
    copy_cols : list (optional)
        Specify the list of columns to copy into the output file.  For a DSP
        file, it should be all columns by default, while for a RAW file, we
        probably do NOT want to copy the waveforms into the output file.
    overwrite : bool
        overwrite existing output files
    evt_tb_name : str (optional)
        Specify the name of the output table instead of automatically setting it.
    builder_config : dict (optional)
        Give a dict of kwargs for `cluster_events`, including coincidence window
        times, conversion to seconds, and names of timestamp and channel columns.

    Returns
    -------
    tcm : pd.Dataframe (conditional!)
        If f_evt is not set, don't write to LH5 file, and instead return a
        DataFrame.
    """
    t_start = time.time()

    # get a list of LH5 tables to decode in the input file
    if lh5_tables is None:
        lh5_tables, lh5_cols = [], {}

        def find_tables(name, obj):
            for attr, val in obj.attrs.items():
                if 'table' in val and 'waveform' not in name: # ignore waveform tables
                    lh5_tables.append(name)
                    col_list = val.replace('table{', '').replace('}','').split(',')
                    if 'waveform' in col_list: col_list.remove('waveform')
                    lh5_cols[name] = col_list

        with h5py.File(f_hit) as hf:
            # to iterate through all groups, you have to pass 'visititems' a function
            hf.visititems(find_tables)

        # for now, assume all column names are the same!
        if copy_cols is None:
            tb_0 = list(lh5_cols.keys())[0]
            copy_cols = lh5_cols[tb_0]

    # get event builder options
    if builder_config is None:
        builder_config = {
        'ts_unit' : 1e-8,       # give the conversion of timestamps to seconds
        'ch_col' : 'channel',   # name of column with channel ID (should be int)
        'ts_col' : 'timestamp', # name of column with timestamps
        'coin_window' : 4e-6,   # length of coincidence window in seconds
        'data_cols' : copy_cols # columns to copy over into output table
        }

    # show status before running
    print('Tables to merge:\n', lh5_tables)
    print('Event builder config:')
    print(builder_config)
    exit()

    # read LH5 tables and convert to DataFrame.  This copy and conversion is
    # a required step!  (Clint will be happy to discuss the reasons with you.)
    print('Loading dataframes ...')
    dfs = []
    for tb in lh5_tables:
        df = lh5.load_dfs(f_hit, copy_cols, lh5_group=tb, idx_list=None, verbose=False)
        print(f'Table: {tb}, rows:{len(df)}')
        dfs.append(df)
        # print(df)

    # # sanity check - test cluster_events with an extra fake channel, #7, with slightly changed timestamps
    # from scipy.ndimage.filters import gaussian_filter
    # df_extra = dfs[1].copy()
    # df_extra['timestamp'] = gaussian_filter(df_extra['timestamp'], sigma=0.1)
    # df_extra['channel'] = 7
    # dfs.append(df_extra)

    # run cluster_events
    print('Building time coincidence map ...')
    tcm = cluster_events(dfs, **builder_config)
    print('Done.  TCM output columns:')
    print(tcm.columns)

    # examine tcm
    # tcm_view = ['channel','tcm_sec','tcm_dt','ix_evt','ix_hit']
    # pd.options.display.max_rows = 50
    # pd.options.display.float_format = '{:,.3e}'.format
    # print(tcm[tcm_view].head(50))
    # print(tcm)
    # print(tcm.columns)

    # write to file if f_evt is set, or return in-memory DataFrame if it isn't.
    if f_evt is None:
        return tcm

    print('Writing to file:', f_evt)
    if os.path.exists(f_evt):
        os.remove(f_evt)

    sto = lh5.LH5Store()
    col_dict = {col : lh5.Array(tcm[col].values, attrs={'units':''}) for col in tcm.columns}
    tb_tcm = lh5.Table(size=len(tcm), col_dict=col_dict)
    tb_name = 'events'
    sto.write_object(tb_tcm, tb_name, f_evt)

    # # sanity check, read it back in and convert back to DataFrame
    # df_check = lh5.load_dfs(f_evt, tcm.columns, lh5_group='events', idx_list=None, verbose=False)
    # print(tcm)
    # print(df_check)

    t_elap = time.time() - t_start
    print(f'Done!  Time elapsed: {t_elap:.2f} sec.')




if __name__=='__main__':
    doc = """Demonstrate usage of the `build_evt` function, to build a time
    coincidence map and organize data from many channels in a single cycle file,
    into an event-like structure where coincindent events are time-ordered and
    grouped together into sub-events."""

    rthf = argparse.RawTextHelpFormatter
    par = argparse.ArgumentParser(description=doc, formatter_class=rthf)
    arg, st, sf = par.add_argument, 'store_true', 'store_false'
    arg('input', type=str, help='input file name (required)')
    arg('-o', '--output', type=str, help='output file name')
    args = par.parse_args()

    # test file -- deleteme
    # /global/project/projectdirs/legend/data/lngs/l200_commissioning/orca_data/pygama/raw/cal/p00/r001/L60_p0_r1_20220606T015344Z_f0.lh5

    # set i/o
    f_in = args.input
    f_out = None if not args.output else args.output

    # run build_evt
    build_evt(f_in, f_out)
