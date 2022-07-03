#!/usr/bin/env python3
import os
import json
import numpy as np
import argparse
import pandas as pd
from pprint import pprint
from collections import OrderedDict

from pygama import DataGroup
from pygama.io.daq_to_raw import daq_to_raw
from pygama.io.raw_to_dsp import raw_to_dsp


def main():
    doc="""
    """
    rthf = argparse.RawTextHelpFormatter
    par = argparse.ArgumentParser(description=doc, formatter_class=rthf)
    arg, st, sf = par.add_argument, 'store_true', 'store_false'
    
    # declare datagroup
    arg('--dg', action=st, help='load datagroup')
    
    # routines
    arg('--d2r', action=st, help='run daq_to_raw')
    arg('--r2d', action=st, help='run raw_to_dsp')
    
    # options
    arg('-o', '--over', action=st, help='overwrite existing files')
    arg('-n', '--nwfs', nargs='*', type=int, help='limit num. waveforms')
    arg('-v', '--verbose', action=st, help='verbose mode')
    
    args = par.parse_args()
    
    expDB = '$LEGEND_META/analysis/LPGTA/LPGTA.json'
    

    # -- set options -- 
    
    nwfs = args.nwfs[0] if args.nwfs is not None else np.inf
    
    print('Processing settings:'
          '\n$LPGTA_DATA =', os.environ.get('LPGTA_DATA'),
          '\n$LEGEND_META =', os.environ.get('LEGEND_META'),
          f'\n  overwrite? {args.over}'
          f'\n  limit wfs? {nwfs}')
    
    # -- run routines -- 
    if args.dg: 
        dg = load_datagroup()
    
    if args.d2r: d2r(dg, args.over, nwfs, args.verbose)
    if args.r2d: r2d(dg, args.over, nwfs, args.verbose)
    
    
def load_datagroup():
    """
    """
    # # -- HADES mode -- 
    # dg = DataGroup('HADES.json')
    # dg.load_df('HADES_fileDB.h5')
    # 
    # # get the first 3 cycle files for det 60A, first th scan
    # que = "detSN=='I02160A' and scantype=='th_HS2_top_psa' and run==1"
    # 
    # # det 60A, lat th scan
    # # que = "detSN=='I02160A' and scantype=='th_HS2_lat_psa'"
    # 
    # # det 60B, first th scan
    # # que = "detSN=='I02160B' and scantype=='th_HS2_top_psa'"
    # 
    # dg.fileDB.query(que, inplace=True)
    # dg.fileDB = dg.fileDB[:3]
    
    
    # # -- CAGE mode -- 
    # dg = DataGroup('CAGE.json')
    # dg.load_df('CAGE_fileDB.h5')
    # que = 'run==8'
    # dg.fileDB.query(que, inplace=True)
    
    
    # # -- LPGTA mode -- 
    # dg = DataGroup('LPGTA.json')
    # dg.load_df('LPGTA_fileDB.h5')
    # # process one big cal file (64 GB)
    # que = "run==18 and YYYYmmdd == '20200302' and hhmmss == '184529'"
    # dg.fileDB.query(que, inplace=True)
    
    # print('files to process:')
    # print(dg.fileDB)
    
    # -- SURF mode -- 
    dg = DataGroup('SURFCHAR.json')
    dg.load_df('SURFCHAR_fileDB.h5')
    
    # can add other filters here
    dg.fileDB = dg.fileDB[:2]
    
    
    return dg
        
    
def d2r(dg, overwrite=False, nwfs=None, vrb=False):
    """
    run daq_to_raw on the current DataGroup
    """
    # print(dg.fileDB)
    # print(dg.fileDB.columns)
    
    subs = dg.subsystems # can be blank: ['']
    # subs = ['geds'] # TODO: ignore other datastreams
    # chans = ['g035', 'g042'] # TODO: select a subset of detectors
    
    print(f'Processing {dg.fileDB.shape[0]} files ...')

    for i, row in dg.fileDB.iterrows():
        
        f_daq = f"{dg.daq_dir}/{row['daq_dir']}/{row['daq_file']}"
        f_raw = f"{dg.lh5_dir}/{row['raw_path']}/{row['raw_file']}"
        subrun = row['cycle'] if 'cycle' in row else None
        
        daq_to_raw(f_daq, f_raw, config=dg.config, subsystems=subs, verbose=vrb,
                   n_max=nwfs, overwrite=overwrite, subrun=subrun)#, chans=chans)
        
        
def r2d(dg, overwrite=False, nwfs=None, vrb=False):
    """
    """
    # print(dg.fileDB)
    # print(dg.fileDB.columns)
    
    with open(f'{dg.experiment}_dsp.json') as f:
        dsp_config = json.load(f, object_pairs_hook=OrderedDict)

    for i, row in dg.fileDB.iterrows():
    
        f_raw = f"{dg.lh5_dir}/{row['raw_path']}/{row['raw_file']}"
        f_dsp = f"{dg.lh5_dir}/{row['dsp_path']}/{row['dsp_file']}"
        
        if "sysn" in f_raw:
            tmp = {'sysn' : 'geds'} # hack for lpgta
            f_raw = f_raw.format_map(tmp)
            f_dsp = f_dsp.format_map(tmp)
        
        raw_to_dsp(f_raw, f_dsp, dsp_config, n_max=nwfs, verbose=vrb,
                   overwrite=overwrite)    
        

    
    
    
if __name__=="__main__":
    main()
    