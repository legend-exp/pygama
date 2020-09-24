#!/usr/bin/env python3
import os.path
import pandas as pd
import numpy as np
from pygama import DataGroup

def main():
    """
    Requires LPGTA.json config file. 
    Save an hdf5 file with pygama daq, raw, and dsp names + paths.
    """
    dg = DataGroup('LPGTA.json')
    
    # dg.lh5_dir_setup() # <-- run this once with create=True

    dg.scan_daq_dir()
    
    # -- organize and augment the dg.file_keys DataFrame -- 
    
    # run 1 & 2 files don't match template
    dg.file_keys.query('run > 2', inplace=True) 
    
    dg.file_keys.sort_values(['run','YYYYmmdd','hhmmss'], inplace=True)
    dg.file_keys.reset_index(drop=True, inplace=True)
    
    def get_cmap(row):
        run_str = f"{row['run']:0>4d}"
        if run_str not in dg.runDB:
            print("Warning: no runDB entry for run", run_str)
            row['cmap'] = ''
            return row
        row['cmap'] = dg.runDB[f"{row['run']:0>4d}"]["cmap"]
        return row
    
    dg.file_keys = dg.file_keys.apply(get_cmap, axis=1)
    
    dg.file_keys['runtype'] = dg.file_keys['rtp']
    
    dg.get_lh5_cols()

    # append a column of filesizes
    sizes_GB = []
    proc_groups = []
    proc_thresh_GB = 128
    proc_LL_GB = 4
    size_sum = 0
    proc_group = 0
    df = dg.file_keys
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
    dg.save_df('LPGTA_fileDB.h5')
    
    print(dg.file_keys)
        
    
if __name__=='__main__':
    main()

