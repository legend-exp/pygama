#!/usr/bin/env python3
import os.path
import argparse
import pandas as pd
import numpy as np

from pygama import DataGroup
import pygama.io.lh5 as lh5

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from tqdm import tqdm
    tqdm.pandas()


def main():
    doc="""
    Post-GERDA Test (PGT): fileDB setup app.
    Creates an HDF5 file with pygama daq/raw/dsp file names & paths.
    Requires: 
    - LPGTA.json config file.
    - env var $LPGTA_DATA = /global/cfs/cdirs/m2676/data/lngs/pgt
    """
    rthf = argparse.RawTextHelpFormatter
    par = argparse.ArgumentParser(description=doc, formatter_class=rthf)
    arg, st, sf = par.add_argument, 'store_true', 'store_false'

    arg('--create_dirs', action=st, help='run only once: create LH5 dirs')
    arg('--init_db', action=st, help='initialize primary ecal output file')
    arg('--runtime', action=st, help='add runtime col to fileDB (use raw file)')
    # arg('--update', action=st, help='update fileDB with new files')
    arg('--show_db', action=st, help='print current fileDB')
    
    args = par.parse_args()

    # load datagroup and show status
    dg = DataGroup('LPGTA.json')
    print('LPGTA_DATA:', os.environ.get('LPGTA_DATA'))
    
    # run routines
    if args.show_db: show_db(dg)
    if args.create_dirs: create_dirs(dg)
    if args.init_db: init_fileDB(dg)
    if args.runtime: get_runtimes(dg)
    

def show_db(dg):
    """
    $ ./setup.py --show_db
    """
    dg.load_df()
    print(dg.fileDB)
    print(dg.fileDB.columns)
    

def create_dirs(dg):
    """
    $ ./setup.py --create_dirs
    only need to run this once
    """
    dg.lh5_dir_setup(create=True)
    

def init_fileDB(dg):
    """
    $ ./setup.py --init_db
    scan daq directory, and augment with additional columns
    """
    dg.scan_daq_dir()
    
    # -- organize and augment the dg.fileDB DataFrame -- 
    
    # run 1 & 2 files don't match template
    dg.fileDB.query('run > 2', inplace=True) 
    
    dg.fileDB.sort_values(['run','YYYYmmdd','hhmmss'], inplace=True)
    dg.fileDB.reset_index(drop=True, inplace=True)
    
    def get_cmap(row):
        run_str = f"{row['run']:0>4d}"
        if run_str not in dg.runDB:
            print("Warning: no runDB entry for run", run_str)
            row['cmap'] = ''
            return row
        row['cmap'] = dg.runDB[f"{row['run']:0>4d}"]["cmap"]
        return row
    
    dg.fileDB = dg.fileDB.apply(get_cmap, axis=1)
    
    dg.fileDB['runtype'] = dg.fileDB['rtp']
    
    dg.get_lh5_cols()

    # add columns for file size and processing group
    proc_thresh_GB = 128
    proc_LL_GB = 4 # lower limit
    sizes_GB, proc_groups = [], []
    size_sum, proc_group = 0, 0
    df = dg.fileDB
    for daq_path, daq_file in zip(df['daq_dir'], df['daq_file']):
        filename = os.path.expandvars('$LPGTA_DATA/daq')+daq_path+'/'+daq_file
        sizes_GB.append(os.path.getsize(filename) / (1024**3))
        # if this file pushes over the threshold, put in a new group
        if size_sum + sizes_GB[-1] > proc_thresh_GB and size_sum > proc_LL_GB: 
            proc_group += 1
            size_sum = 0
        proc_groups.append(proc_group)
        size_sum += sizes_GB[-1]
        if size_sum > proc_thresh_GB:
            proc_group += 1
            size_sum = 0
    df['daq_size_GB'] = sizes_GB
    df['proc_group'] = proc_groups
    
    # save to file used by processing.py
    print(dg.fileDB)
    print(dg.fileDB.columns)
    print('FileDB location:', dg.config['fileDB'])
    ans = input('Save new fileDB? (y/n)')
    if ans.lower() == 'y':
        dg.save_df(dg.config['fileDB'])
    
    
def get_runtimes(dg):
    """
    $ ./setup.py --runtime
    
    Get the Ge runtime of each cycle file (in seconds).  
    Add a 'ge_runtime' column to the fileDB.
    Requires the raw LH5 files.
    """
    dg.load_df()
    # dg.fileDB = dg.fileDB[50:55] # debug only
    
    # reset columns of interest
    new_cols = ['runtime', 'rt_std']
    for col in new_cols:
        if col in dg.fileDB.columns:
            dg.fileDB.drop(col, axis=1, inplace=True)
    
    sto = lh5.Store()
    
    t_start = time.time()
    def runtime_cycle(df_row):
        
        # load raw file path (with {these} in it)
        f_raw = f'{dg.lh5_dir}/{df_row.raw_path}/{df_row.raw_file}'
        f_raw = f_raw.format_map({'sysn':'geds'})
        
        # always look for Ge
        f_key = df_row.raw_file.format_map({'sysn':'geds'})
        if not os.path.exists(f_raw):
            # print(f'no Ge data: {f_key}')
            return pd.Series({'runtime':0, 'rt_std':0})
        
        # for PGT, compare the first three channels (for redundancy)
        rts = []
        ge_groups = sto.ls(f_raw)
        for ge in ge_groups[:3]:
            ts = lh5.load_nda([f_raw], ['timestamp'], ge+'/raw/')['timestamp']
            rts.append(ts[-1])
        
        # take largest value & compute uncertainty
        runtime = max(rts) / 60
        rt_std = np.std(np.array([rts]))
        # print(f_key, runtime, rt_std)
        
        return pd.Series({'runtime':runtime, 'rt_std':rt_std})
        
    # df_tmp = dg.fileDB.apply(runtime_cycle, axis=1)
    dg.fileDB[new_cols] = dg.fileDB.progress_apply(runtime_cycle, axis=1)
    
    print(f'Done. Time elapsed: {(time.time()-t_start)/60:.2f} mins.')
    
    # save to fileDB if everything looks OK
    print(dg.fileDB)
    print(dg.fileDB.columns)
    print('FileDB location:', dg.config['fileDB'])
    ans = input('Save new fileDB? (y/n)')
    if ans.lower() == 'y':
        dg.save_df(dg.config['fileDB'])
    
    
if __name__=='__main__':
    main()

