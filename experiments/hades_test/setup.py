#!/usr/bin/env python3
import os, sys
import argparse
import pandas as pd
import numpy as np
from pprint import pprint

from pygama import DataGroup
from pygama.io.orcadaq import parse_header
import pygama.io.lh5 as lh5

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from tqdm import tqdm
    tqdm.pandas() # suppress annoying FutureWarning

def main():
    doc="""
    Create and maintain the 'fileDB' needed by DataGroup.
    Provides options for first-time setup, and updating an existing fileDB.
    """
    rthf = argparse.RawTextHelpFormatter
    par = argparse.ArgumentParser(description=doc, formatter_class=rthf)
    arg, st, sf = par.add_argument, 'store_true', 'store_false'

    # initial setup
    arg('--mkdirs', action=st, help='run first-time directory setup')
    arg('--init', action=st, help='run first-time DAQ directory scan')

    # update mode (normal)
    arg('-u', '--update', action=st, help='rescan DAQ dir, update existing fileDB')
    arg('--orca', action=st, help='scan ORCA XML headers of DAQ files')
    arg('--rt', action=st, help='get runtimes (requires dsp file)')

    # options
    arg('-b', '--batch', action=st, help='batch mode, do not ask for user y/n')
    arg('--show', action=st, help='show current on-disk fileDB')
    arg('-o', '--over', action=st, help='overwrite existing fileDB')
    arg('--lh5_user', action=st, help='use $CAGE_LH5_USER over $CAGE_LH5')

    args = par.parse_args()

    # declare main DataGroup
    dg = DataGroup('hades.json')

    # -- run routines --
    if args.mkdirs: dg.lh5_dir_setup(args.lh5_user)
    if args.show: show_fileDB(dg)
    if args.init: init(dg)
    if args.update: update(dg, args.batch)
    
    # not for flashcam
    # if args.orca: scan_orca_headers(dg, args.over, args.batch)
    
    # need some other way to do this
    # if args.rt: get_runtimes(dg, args.over, args.batch)


def show_fileDB(dg):
    """
    $ ./setup.py --show
    Show current on-disk fileDB.

    Columns:
    - Added by get_cyc_info: 'YYYY', 'cycle', 'daq_dir', 'daq_file', 'dd', 'mm',
                             'run', 'runtype', 'unique_key'
    - Added by get_lh5_cols: 'raw_file', 'raw_path', 'dsp_file', 'dsp_path',
                             'hit_file', 'hit_path'
    - Added by scan_orca_headers: 'startTime', 'threshold',
    - Added by get_runtime: 'stopTime', 'runtime'
    """
    dg.load_df()

    dbg_cols = ['run', 'cycle', 'unique_key']

    if 'startTime' in dg.fileDB.columns:
        dbg_cols += ['startTime']

    if 'runtime' in dg.fileDB.columns:
        dbg_cols += ['runtime']

    print(dg.fileDB[dbg_cols])
    print(dg.fileDB.columns)


def init(dg):
    """
    ./setup.py --init
    Run first scan of the fileDB over the DAQ directory ($CAGE_DAQ)
    """
    print('Initializing fileDB...')
    
    # scan over DAQ directory, then organize by cycle (ORCA run number)
    dg.scan_daq_dir()
    dg.fileDB.sort_values(['cycle'], inplace=True)
    dg.fileDB.reset_index(drop=True, inplace=True)
    dg.fileDB = dg.fileDB.apply(get_cyc_info, args=[dg], axis=1)

    # compute lh5 column names (uses $CAGE_LH5, don't use $CAGE_LH5_USER here)
    dg.get_lh5_cols()

    # attempt to convert to integer (will fail if nan's are present)
    for col in ['run', 'cycle']:
        dg.fileDB[col] = pd.to_numeric(dg.fileDB[col])

    print(dg.fileDB[['run', 'cycle', 'daq_file', 'runtype', 'skip']].to_string())

    print('Ready to save.  This will overwrite any existing fileDB.')
    ans = input('Continue? (y/n) ')
    if ans.lower() == 'y':
        dg.save_df(dg.config['fileDB'])
        print('Wrote fileDB:', dg.config['fileDB'])


def update(dg, batch_mode=False):
    """
    ./setup.py -u
    After taking new data, run this function to add rows to fileDB.
    New rows will not have all columns yet.
    TODO: look for nan's to identify cycles not covered in runDB
    """
    print('Updating fileDB ...')
    
    dbg_cols = ['unique_key', 'run', 'cycle', 'daq_file']

    # load existing file keys
    dg.load_df()
    # print(dg.fileDB[dbg_cols])

    # scan daq dir for new file keys
    dg_new = DataGroup('cage.json')
    dg_new.scan_daq_dir()
    dg_new.file_keys.sort_values(['cycle'], inplace=True)
    dg_new.file_keys.reset_index(drop=True, inplace=True)
    dg_new.file_keys = dg_new.file_keys.apply(get_cyc_info, args=[dg_new], axis=1)
    dg_new.get_lh5_cols()
    for col in ['run', 'cycle']:
        dg_new.file_keys[col] = pd.to_numeric(dg_new.file_keys[col])
    # print(dg_new.file_keys[dbg_cols])

    # identify new keys, save new indexes
    df1 = dg.fileDB['unique_key']
    df2 = dg_new.file_keys['unique_key']
    new_keys = pd.concat([df1, df2]).drop_duplicates(keep=False)
    new_idx = new_keys.index

    if len(new_keys) > 0:
        print('Found new files:')
        print(new_keys)

        print('Merging with existing fileDB:')
        df_upd = pd.concat([dg.fileDB, dg_new.file_keys.loc[new_idx]])
        print(df_upd[dbg_cols])

        if not batch_mode:
            print("RunDB Check -- did you update runDB.json?  Are there any NaN's in filenames/paths above?")
            ans = input('Save updated fileDB? (y/n):')
            if ans.lower() == 'y':
                dg.fileDB = df_upd
                dg.save_df(dg.config['fileDB'])
                print('fileDB updated.')
        else:
            dg.fileDB = df_upd
            dg.save_df(dg.config['fileDB'])
            print('fileDB updated.')
    else:
        print('No new files found!  current fileDB:')
        print(dg.fileDB[dbg_cols])


def get_cyc_info(row, dg):
    """
    using the runDB, map cycle numbers to physics runs, identify detector,
    physics run type, etc.
    """
    # loop over the runDB and add columns to each row of dg.fileDB
    cyc = row['cycle']
    for run, cycles in dg.runDB.items():
        tmp = cycles[0].split(',')
        for rng in tmp:
            if '-' in rng:
                clo, chi = [int(x) for x in rng.split('-')]
                if clo <= cyc <= chi:
                    row['run'] = run
                    row['runtype'] = cycles[1]
                    break
            else:
                clo = int(rng)
                if cyc == clo:
                    row['run'] = run
                    row['runtype'] = cycles[1]
                    break

    # label the detector
    if 0 < cyc <= 124:
        row['detector'] = 'oppi_v1'
    elif 125 <= cyc <= 136:
        row['detector'] = 'icpc_v1'
    elif 137 <= cyc <= 9999:
        row['detector'] = 'oppi_v2'
        
    # apply file selection
    skips = dg.runSelectionDB['daq_junk_cycles']
    row['skip'] = cyc in skips
    
    return row


def scan_orca_headers(dg, overwrite=False, batch_mode=False):
    """
    $ ./setup.py --orca
    Add unix start time, threshold, and anything else in the ORCA XML header
    for the fileDB.
    """
    print('Scanning ORCA headers ...')
    
    # load existing fileDB
    dg.load_df()

    # first-time setup
    if 'startTime' not in dg.fileDB.columns or overwrite:
        df_keys = dg.fileDB.copy()
        update_existing = False
        print('Re-scanning entire fileDB')

    elif 'startTime' in dg.fileDB.columns:
        # look for any rows with nans to update
        idx = dg.fileDB.loc[pd.isna(dg.fileDB['startTime']), :].index
        if len(idx) > 0:
            df_keys = dg.fileDB.loc[idx].copy()
            print(f'Found {len(df_keys)} new files without startTime:')
            print(df_keys)
            update_existing = True
        else:
            print('No empty startTime values found.')
            df_keys = pd.DataFrame()

    if len(df_keys) == 0:
        print('No files to update.  Exiting...')
        exit()

    # clear new colums if they exist
    new_cols = ['startTime', 'threshold', 'daq_gb']
    for col in new_cols:
        if col in df_keys.columns:
            df_keys.drop(col, axis=1, inplace=True)

    def scan_orca_header(df_row):
        f_daq = dg.daq_dir + df_row['daq_dir'] + '/' + df_row['daq_file']
        daq_gb = os.path.getsize(f_daq) / 1e9

        if not os.path.exists(f_daq) and not df_row.skip:
            print(f"Error, file doesn't exist:\n  {f_daq}")
            exit()
        if df_row['skip']==True:
            print(f"Skipping cycle: {df_row['cycle']}")
            return pd.Series({'startTime':np.nan, 'threshold':np.nan, 
                              'daq_gb':daq_gb})
        else:
            _,_, header_dict = parse_header(f_daq)
            # pprint(header_dict)
            info = header_dict['ObjectInfo']
            t_start = info['DataChain'][0]['Run Control']['startTime']
            thresh = info['Crates'][0]['Cards'][1]['thresholds'][2]
            return pd.Series({'startTime':t_start, 'threshold':thresh,
                              'daq_gb':daq_gb})

    df_tmp = df_keys.progress_apply(scan_orca_header, axis=1)
    df_keys[new_cols] = df_tmp

    if update_existing:
        idx = dg.fileDB.loc[pd.isna(dg.fileDB['startTime']), :].index
        dg.fileDB.loc[idx] = df_keys
    else:
        dg.fileDB = df_keys

    dbg_cols = ['run', 'cycle', 'daq_file', 'startTime', 'daq_gb']
    print(dg.fileDB[dbg_cols])

    print('Ready to save.  This will overwrite any existing fileDB.')
    if not batch_mode:
        ans = input('Save updated fileDB? (y/n):')
        if ans.lower() == 'y':
            dg.save_df(dg.config['fileDB'])
            print('fileDB updated.')
    else:
        dg.save_df(dg.config['fileDB'])
        print('fileDB updated.')


def get_runtimes(dg, overwrite=False, batch_mode=False):
    """
    $ ./setup.py --rt
    
    Compute runtime (# minutes in run) and stopTime (unix timestamp) using
    the timestamps in the DSP file.
    
    NOTE: Could change this to use the raw file timestamps instead of dsp file, 
          but that still makes this function dependent on a processing step.
    NOTE: CAGE uses struck channel 2 (0-indexed)
    """
    print('Scanning DSP files for runtimes ...')
    
    # load existing fileDB
    dg.load_df()

    # first-time setup
    if 'runtime' not in dg.fileDB.columns or overwrite:
        df_keys = dg.fileDB.copy()
        update_existing = False
        print('Re-scanning entire fileDB')

    elif 'runtime' in dg.fileDB.columns:
        # look for any rows with nans to update
        idx = dg.fileDB.loc[pd.isna(dg.fileDB['runtime']), :].index
        if len(idx) > 0:
            df_keys = dg.fileDB.loc[idx].copy()
            print(f'Found {len(df_keys)} new files without runtime:')
            print(df_keys)
            update_existing = True
        else:
            print('No empty runtime values found.')

    if len(df_keys) == 0:
        print('No files to update.  Exiting...')
        exit()

    # clear new colums if they exist
    new_cols = ['stopTime', 'runtime']
    for col in new_cols:
        if col in df_keys.columns:
            df_keys.drop(col, axis=1, inplace=True)

    sto = lh5.Store()
    def get_runtime(df_row):

        # load timestamps from dsp file
        f_dsp = dg.lh5_dir + df_row['dsp_path'] + '/' + df_row['dsp_file']

        if not os.path.exists(f_dsp) and not df_row.skip:
            print(f"Error, file doesn't exist:\n  {f_dsp}")
            exit()
        elif df_row.skip:
            print(f'Skipping cycle file:\n  {f_dsp}')
            return pd.Series({'stopTime':0, 'runtime':0})

        data, n_rows = sto.read_object('ORSIS3302DecoderForEnergy/dsp', f_dsp)

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

    if update_existing:
        idx = dg.fileDB.loc[pd.isna(dg.fileDB['runtime']), :].index
        dg.fileDB.loc[idx] = df_keys
    else:
        dg.fileDB = df_keys

    dbg_cols = ['run', 'cycle', 'unique_key', 'startTime', 'runtime']
    print(dg.fileDB[dbg_cols])

    print('Ready to save.  This will overwrite any existing fileDB.')
    if not batch_mode:
        ans = input('Save updated fileDB? (y/n):')
        if ans.lower() == 'y':
            dg.fileDB = df_keys
            dg.save_df(dg.config['fileDB'])
            print('fileDB updated.')
    else:
        dg.fileDB = df_keys
        dg.save_df(dg.config['fileDB'])
        print('fileDB updated.')


if __name__=='__main__':
    main()
