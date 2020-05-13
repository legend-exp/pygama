#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pprint import pprint
from pygama import DataGroup

def main():
    """
    """
    # analyze_lpgta()
    # analyze_cage()
    analyze_hades()
    # analyze_ornl()
    

def analyze_lpgta():
    
    dg = DataGroup('LPGTA.json')
    
    # dg.lh5_dir_setup()
    dg.scan_daq_dir()
    
    # -- experiment-specific choices -- 
    
    # run 1 & 2 files don't match template
    dg.file_keys.query('run > 2', inplace=True) 
    
    dg.file_keys.sort_values(['run','YYYYmmdd','hhmmss'], inplace=True)
    dg.file_keys.reset_index(drop=True, inplace=True)
    
    def get_cmap(row):
        row['cmap'] = dg.runDB[f"{row['run']:0>4d}"]["cmap"]
        return row
    
    dg.file_keys = dg.file_keys.apply(get_cmap, axis=1)
    
    print(dg.file_keys)
    
    # dg.save_keys()
    # dg.load_keys()
    
    dg.save_df('./LPGTA_fileDB.h5')
    dg.load_df('./LPGTA_fileDB.h5')
    
    print(dg.file_keys)
    
    
def analyze_cage():
    
    dg = DataGroup('CAGE.json')
    dg.lh5_dir_setup()
    
    # dg.scan_daq_dir()
    # 
    # # -- experiment-specific choices -- 
    # dg.file_keys.sort_values(['cycle'], inplace=True)
    # dg.file_keys.reset_index(drop=True, inplace=True)
    # 
    # def get_cyc_info(row):
    #     """ 
    #     map cycle numbers to physics runs, and identify detector 
    #     """
    #     cyc = row['cycle']
    #     for run, cycles in dg.runDB.items():
    #         tmp = cycles[0].split(',')
    #         for rng in tmp:
    #             if '-' in rng:
    #                 clo, chi = [int(x) for x in rng.split('-')]
    #                 if clo <= cyc <= chi:
    #                     row['run'] = run
    #                     break
    #             else:
    #                 clo = int(rng)
    #                 if cyc == clo:
    #                     row['run'] = run
    #                     break
    #     # label the detector ('runtype' matches 'run_types' in config file)
    #     if cyc < 126:
    #         row['runtype'] = 'oppi'
    #     else:
    #         row['runtype'] = 'icpc'
    #     return row
    # 
    # dg.file_keys = dg.file_keys.apply(get_cyc_info, axis=1)
    
    # dg.save_keys()
    # dg.load_keys()
    # print(dg.file_keys)
    
    # dg.save_df('CAGE_fileDB.h5')
    # exit()

    dg.load_df('CAGE_fileDB.h5')
    # print(dg.file_keys)
    
    dg.get_lh5_cols()
    
    
def analyze_hades():
    """
    """
    
    dg = DataGroup('HADES.json')
    # dg.lh5_dir_setup()
    # dg.scan_daq_dir()
    
    # dg.save_keys()
    # dg.load_keys() # this is really slow
    
    # -- experiment-specific stuff -- 
    # dg.file_keys['runtype'] = dg.file_keys['detSN']
    
    # sort by timestamp
    # dg.save_df('HADES_fileDB.h5')
    dg.load_df('HADES_fileDB.h5')
    
    # andreas request: show how to group files in the same run

    dg.file_keys = dg.file_keys.query("detSN=='I02160B' and scantype=='ba_HS4_top_dlt' and run==1")
    
    # do a sort by timestamp
    def get_ts(row):
        ts = str(row['YYmmdd']) + str(row['hhmmss'])
        row['date'] = pd.to_datetime(ts, format='%Y%m%d%H%M%S')
        return row
    dg.file_keys = dg.file_keys.apply(get_ts, axis=1)
    dg.file_keys.sort_values('date', inplace=True)
    
    
    # dg.get_lh5_cols()
    # print(dg.file_keys)
    
    
    
    
def analyze_ornl():
    
    dg = DataGroup('ORNL.json')
    # dg.lh5_dir_setup()
    dg.scan_daq_dir()
    
    # expt-specific organization
    dg.file_keys.sort_values(['cycle'], inplace=True)
    dg.file_keys.reset_index(drop=True, inplace=True)

    dg.save_keys()
    dg.load_keys()
    print(dg.file_keys)
    
    
if __name__=='__main__':
    main()
    
    