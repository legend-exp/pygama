#!/usr/bin/env python3
import os, sys
import pandas as pd
import numpy as np
from pprint import pprint

from pygama import DataGroup
from pygama.io.orcadaq import parse_header
import pygama.lh5 as lh5

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from tqdm import tqdm
    tqdm.pandas() # suppress annoying FutureWarning


def main():
    """
    """
    dg = DataGroup('oppi.json')
    
    # init(dg) # only run first time
    # update(dg) 
    # scan_orca_headers(dg)
    # get_runtimes(dg) # requires dsp file right now (at least raw)
    
    # show_dg(dg)


def init(dg):
    """
    Initial setup of the fileDB.
    """
    dg.lh5_dir_setup(create=True)
    dg.scan_daq_dir()

    dg.fileDB.sort_values(['cycle'], inplace=True)
    dg.fileDB.reset_index(drop=True, inplace=True)
    dg.fileDB = dg.fileDB.apply(get_cyc_info, args=[dg], axis=1)
    dg.get_lh5_cols()
    
    for col in ['run', 'cycle']:
        dg.fileDB[col] = pd.to_numeric(dg.fileDB[col])
    # dg.fileDB['run'] = dg.fileDB['run'].astype(int)

    print(dg.fileDB[['run', 'cycle', 'unique_key', 'daq_file']].to_string())
    dg.save_df(dg.config['fileDB'])
    print('Wrote output file.')


def update(dg):
    """
    After taking new data, run this function to add rows to fileDB.
    New rows will not have all columns yet
    """
    df_scan = dg.scan_daq_dir(update=True)

    df_scan.sort_values(['cycle'], inplace=True)
    df_scan.reset_index(drop=True, inplace=True)
    df_scan = df_scan.apply(get_cyc_info, args=[dg], axis=1)
    df_scan = dg.get_lh5_cols(update_df=df_scan)
    for col in ['run', 'cycle']:
        df_scan[col] = pd.to_numeric(df_scan[col])
        
    # look for nan's to identify cycles not covered in runDB

    # merge the new df into the existing one based on unique key
    df_keys = pd.read_hdf(dg.config['fileDB'], key='file_keys')
    print(df_scan)
    print(df_keys)
    print('clint, you need to merge by unique_key')
    exit()

    # this might work, but you have to pull out the duplicates first
    # dg.fileDB = pd.concat([df_keys, df_scan])

    print(dg.fileDB)

    ans = input('Save file key DF? y/n')
    if ans.lower() == 'y':
        dg.save_df(dg.config['fileDB'])


def get_cyc_info(row, dg):
    """
    map cycle numbers to physics runs, and identify detector
    """
    myrow = row.copy() # i have no idea why mjcenpa makes me do this
    cyc = myrow['cycle']
    for run, cycles in dg.runDB.items():
        tmp = cycles[0].split(',')
        for rng in tmp:
            if '-' in rng:
                clo, chi = [int(x) for x in rng.split('-')]
                if clo <= cyc <= chi:
                    myrow['run'] = run
                    break
            else:
                clo = int(rng)
                if cyc == clo:
                    myrow['run'] = run
                    break

    # label the detector
    if 0 < cyc <= 2018:
        myrow['runtype'] = 'mj60'
    elif 2019 <= cyc <= 2360:
        myrow['runtype'] = 'oppi'
    return myrow


def scan_orca_headers(dg):
    """
    add runtime and threshold columns to the fileDB.
    to get threshold, read it out of the orca header.
    to get runtime, we have to access the raw timestamps and correct for rollover
    NOTE: we used struck channel 2 (0-indexed)
    """
    write_output = True

    df_keys = pd.read_hdf(dg.config['fileDB'])

    # clear new colums if they exist
    new_cols = ['startTime', 'threshold']
    for col in new_cols:
        if col in df_keys.columns:
            df_keys.drop(col, axis=1, inplace=True)

    def scan_orca_header(df_row):
        f_daq = dg.daq_dir + df_row['daq_dir'] + '/' + df_row['daq_file']
        _,_, header_dict = parse_header(f_daq)
        # pprint(header_dict)
        info = header_dict['ObjectInfo']
        t_start = info['DataChain'][0]['Run Control']['startTime']
        thresh = info['Crates'][0]['Cards'][1]['thresholds'][2]
        return pd.Series({'startTime':t_start, 'threshold':thresh})

    df_tmp = df_keys.progress_apply(scan_orca_header, axis=1)
    df_keys[new_cols] = df_tmp

    if write_output:
        df_keys.to_hdf(dg.config['fileDB'], key='file_keys')
        print(f"Wrote output file: {dg.config['fileDB']}")


def get_runtimes(dg):
    """
    Requires DSP files.
    compute runtime (# minutes in run) and stopTime (unix timestamp) using
    the timestamps in the dsp file.
    """
    write_output = True

    df_keys = pd.read_hdf(dg.config['fileDB'])

    # clear new colums if they exist
    new_cols = ['stopTime', 'runtime']
    for col in new_cols:
        if col in df_keys.columns:
            df_keys.drop(col, axis=1, inplace=True)

    sto = lh5.Store()
    def get_runtime(df_row):

        # load timestamps from dsp file
        f_dsp = dg.lh5_dir + df_row['dsp_path'] + '/' + df_row['dsp_file']
        data = sto.read_object('ORSIS3302DecoderForEnergy/dsp', f_dsp)

        # correct for timestamp rollover
        clock = 100e6 # 100 MHz
        UINT_MAX = 4294967295 # (0xffffffff)
        t_max = UINT_MAX / clock

        # ts = data['timestamp'].nda.astype(np.int64) # must be signed for np.diff
        ts = data['timestamp'].nda / clock # converts to float

        tdiff = np.diff(ts)
        tdiff = np.insert(tdiff, 0 , 0)
        iwrap = np.where(tdiff < 0)
        iloop = np.append(iwrap[0], len(ts))

        ts_new, t_roll = [], 0
        for i, idx in enumerate(iloop):
            ilo = 0 if i==0 else iwrap[0][i-1]
            ihi = idx
            ts_block = ts[ilo:ihi]
            t_last = ts[ilo-1]
            t_diff = t_max - t_last
            ts_new.append(ts_block + t_roll)
            t_roll += t_last + t_diff
        ts_corr = np.concatenate(ts_new)

        # calculate runtime and unix stopTime
        rt = ts_corr[-1] / 60 # minutes
        st = int(np.ceil(df_row['startTime'] + rt * 60))

        return pd.Series({'stopTime':st, 'runtime':rt})

    df_tmp = df_keys.progress_apply(get_runtime, axis=1)
    df_keys[new_cols] = df_tmp

    print(df_keys)

    if write_output:
        df_keys.to_hdf(dg.config['fileDB'], key='file_keys')
        print(f"Wrote output file: {dg.config['fileDB']}")


def show_dg(dg):
    """
    datagroup columns:
    ['YYYY', 'cycle', 'daq_dir', 'daq_file', 'dd', 'mm', 'run', 'runtype',
       'unique_key', 'raw_file', 'raw_path', 'dsp_file', 'dsp_path',
       'hit_file', 'hit_path', 'startTime', 'threshold', 'stopTime',
       'runtime']
    """
    df_keys = pd.read_hdf(dg.config['fileDB'])
    # print(df_keys.columns)

    dbg_cols = ['run', 'cycle', 'unique_key', 'startTime']
    # print(df_keys[dbg_cols].to_string())
    print(df_keys[dbg_cols])



if __name__=='__main__':
    main()
