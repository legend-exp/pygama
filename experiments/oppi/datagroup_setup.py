#!/usr/bin/env python3
import os
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
    """
    Requires oppi.json config file.
    """
    # setup()
    # scan_orca_headers()
    get_runtimes()


def setup():
    """
    Save an hdf5 file with pygama daq, raw, and dsp names + paths.
    """
    dg = DataGroup('oppi.json')
    # dg.lh5_dir_setup(create=True) # <-- run this once with create=True
    dg.scan_daq_dir()

    # -- experiment-specific choices --
    dg.file_keys.sort_values(['cycle'], inplace=True)
    dg.file_keys.reset_index(drop=True, inplace=True)

    def get_cyc_info(row):
        """
        map cycle numbers to physics runs, and identify detector
        """
        cyc = row['cycle']
        for run, cycles in dg.runDB.items():
            tmp = cycles[0].split(',')
            for rng in tmp:
                if '-' in rng:
                    clo, chi = [int(x) for x in rng.split('-')]
                    if clo <= cyc <= chi:
                        row['run'] = run
                        break
                else:
                    clo = int(rng)
                    if cyc == clo:
                        row['run'] = run
                        break
        # label the detector
        row['runtype'] = 'oppi'
        return row

    dg.file_keys = dg.file_keys.apply(get_cyc_info, axis=1)

    dg.get_lh5_cols()

    for col in ['run']:
        dg.file_keys[col] = pd.to_numeric(dg.file_keys[col])

    # -- filter out MJ60 runs --
    dg.file_keys = dg.file_keys.loc[dg.file_keys.run>=0].copy()

    dg.file_keys['run'] = dg.file_keys['run'].astype(int)

    print(dg.file_keys)

    dg.save_df('oppi_fileDB.h5')


def scan_orca_headers():
    """
    add runtime and threshold columns to the fileDB.
    to get threshold, read it out of the orca header.
    to get runtime, we have to access the raw timestamps and correct for rollover
    NOTE: we used struck channel 7 (0-indexed)
    """
    write_output = True
    
    dg = DataGroup('oppi.json')
    df_keys = pd.read_hdf('oppi_fileDB.h5')
    
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
        thresh = info['Crates'][0]['Cards'][1]['thresholds'][7]
        return pd.Series({'startTime':t_start, 'threshold':thresh})
    
    df_tmp = df_keys.progress_apply(scan_orca_header, axis=1)
    df_keys[new_cols] = df_tmp
    
    if write_output:
        df_keys.to_hdf('oppi_fileDB.h5', key='file_keys')
        print('Wrote output file: oppi_fileDB.h5, key: file_keys')
    

def get_runtimes():
    """
    compute runtime (# minutes in run) and stopTime (unix timestamp) using
    the timestamps in the dsp file.
    """
    write_output = True
    
    dg = DataGroup('oppi.json')
    df_keys = pd.read_hdf('oppi_fileDB.h5')
    
    # clear new colums if they exist
    new_cols = ['stopTime', 'runtime']
    for col in new_cols:
        if col in df_keys.columns:
            df_keys.drop(col, axis=1, inplace=True)
    
    sto = lh5.Store()
    def get_runtime(df_row):

        # load timestamps from dsp file
        f_dsp = dg.lh5_dir + df_row['dsp_path'] + '/' + df_row['dsp_file']
        data = sto.read_object('ORSIS3302DecoderForEnergy/raw', f_dsp)

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
        df_keys.to_hdf('oppi_fileDB.h5', key='file_keys')
        print('Wrote output file: oppi_fileDB.h5, key: file_keys')
    

if __name__=='__main__':
    main()
