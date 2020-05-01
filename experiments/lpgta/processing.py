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
    
    # routines
    arg('--d2r', action=st, help='run daq_to_raw')
    
    # options
    arg('--over', action=st, help='overwrite existing files')
    arg('-n', '--nwfs', nargs='*', type=int, help='limit num. waveforms')
    arg('-v', '--verbose', action=st, help='verbose mode')
    
    args = par.parse_args()
    
    expDB = '$LEGEND_META/analysis/LPGTA/LPGTA.json'
    
    # TODO: allow DataGroup to be set by cmd line
    dg = DataGroup(21, config=expDB, nfiles=3)
    
    # -- run routines -- 
    nwfs = args.nwfs[0] if args.nwfs is not None else np.inf
    
    print('Processing settings:'
          '\n  $LPGTA_DATA =', os.environ.get('LPGTA_DATA'),
          '\n  $LEGEND_META =', os.environ.get('LEGEND_META'),
          f'\n  overwrite? {args.over}'
          f'\n  limit wfs? {nwfs}')
    
    if args.d2r: d2r(dg, args.over, nwfs, args.verbose)
    
    
def d2r(dg, overwrite=False, nwfs=None, vrb=False):
    """
    run daq_to_raw on the current DataGroup
    """
    df_daq = dg.find_daq_files()
    
    subs = dg.subsystems
    
    # subs = ['geds'] # ignore other datastreams
    chans = ['g035', 'g042'] # optional: select a subset of detectors

    for i, row in df_daq.iterrows():
        
        f_daq, f_raw = row[['daq_file','raw_file']]
                
        daq_to_raw(f_daq, f_raw, config=dg.config, subsystems=subs, verbose=vrb,
                   n_max=nwfs, chans=chans)
        
        exit()

        
def r2d(dg):
    """
    """
    df_raw = dg.find_raw_files()
    
    for i, row in df_raw.iterrows():
        
        f_raw, f_dsp = row[['raw_file','dsp_file']]
        
        # load LH5 tables
        # init mpi4py
        # calculate table splits
        
        # in each thread:
        # pc = ProcessingChain()
        # pc.init_json()
        raw_to_dsp(table_in, table_out)

    
    
    
if __name__=="__main__":
    main()
    