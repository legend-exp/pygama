#!/usr/bin/env python3
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
    
    # save to file used by processing.py
    dg.save_df('./LPGTA_fileDB.h5')
    
    print(dg.file_keys)
        
    
if __name__=='__main__':
    main()

