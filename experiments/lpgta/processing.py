#!/usr/bin/env python3
import os
import numpy as np
import argparse
import pandas as pd
from pprint import pprint

from pygama import DataGroup
from pygama.io.daq_to_raw import daq_to_raw


def main():
    doc="""
    LPGTA data processing routine. 
    You must set these environment variables:
      * $LPGTA_DATA : base data directory
      * $LEGEND_META : the legend-metadata repository
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
    arg('--over', action=st, help='overwrite existing files')
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
    if args.dg: load_datagroup()
    
    if args.d2r: d2r(dg, args.over, nwfs, args.verbose)
    if args.r2d: r2d(dg, args.over, nwfs, args.verbose)
    
    
def load_datagroup():
    """
    """
    print('hi')
    exit()
    
    daq_path = '/'.join(f for f in df_daq['daq_file'][0].split('/')[:-1])
    raw_path = '/'.join(f for f in df_daq['raw_file'][0].split('/')[:-1])
    print('DAQ path:', daq_path)
    print('RAW path:', raw_path)
    df_daq['daq_file'] = [f.split('/')[-1] for f in df_daq['daq_file']]
    df_daq['raw_file'] = [f.split('/')[-1] for f in df_daq['raw_file']]
    # print(df_daq.columns)
    
    view_cols = ['date','run','YYYYmmdd','hhmmss','rtp','daq_file','raw_file']
    print(df_daq[view_cols].to_string())
        
    
def d2r(dg, overwrite=False, nwfs=None, vrb=False):
    """
    run daq_to_raw on the current DataGroup
    """
    df_daq = dg.find_daq_files()
    
    
    
    
    
    exit()
    
    subs = dg.subsystems
    
    # subs = ['geds'] # TODO: ignore other datastreams
    # chans = ['g035', 'g042'] # TODO: select a subset of detectors
    
    print(f'Processing {df_daq.shape[0]} files ...')

    for i, row in df_daq.iterrows():
        
        f_daq, f_raw = row[['daq_file','raw_file']]
        
        daq_to_raw(f_daq, f_raw, config=dg.config, subsystems=subs, verbose=vrb,
                   n_max=nwfs)#, chans=chans)
        
        
def r2d(dg, overwrite=False, nwfs=None, vrb=False):
    """
    """
    df_raw = dg.find_raw_files()
    
    # for i, row in df_raw.iterrows():
    # 
    #     f_raw, f_dsp = row[['raw_file','dsp_file']]
    # 
    #     # load LH5 tables
    #     # init mpi4py
    #     # calculate table splits
    # 
    #     # in each thread:
    #     # pc = ProcessingChain()
    #     # pc.init_json()
    #     raw_to_dsp(table_in, table_out)

    
    
    
if __name__=="__main__":
    main()
    