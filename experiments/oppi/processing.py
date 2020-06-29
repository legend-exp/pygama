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
    OPPI data processing routine.
    TODO: parallelize, submit processing jobs
    """
    rthf = argparse.RawTextHelpFormatter
    par = argparse.ArgumentParser(description=doc, formatter_class=rthf)
    arg, st, sf = par.add_argument, 'store_true', 'store_false'

    # declare datagroup
    arg('--dg', action=st, help='load datagroup')

    # routines
    arg('--d2r', action=st, help='run daq_to_raw')
    arg('--r2d', action=st, help='run raw_to_dsp')
    arg('--r2d_file', nargs=2, type=str, help='single-file raw_to_dsp')

    # options
    arg('-o', '--over', action=st, help='overwrite existing files')
    arg('-n', '--nwfs', nargs='*', type=int, help='limit num. waveforms')
    arg('-v', '--verbose', action=st, help='verbose mode')

    args = par.parse_args()

    # -- set options --

    nwfs = args.nwfs[0] if args.nwfs is not None else np.inf

    print('Processing settings:'
          # '\n$LPGTA_DATA =', os.environ.get('LPGTA_DATA'),
          # '\n$LEGEND_META =', os.environ.get('LEGEND_META'),
          f'\n  overwrite? {args.over}'
          f'\n  limit wfs? {nwfs}')

    # -- run routines --
    if args.dg:
        dg = load_datagroup()

    if args.d2r: d2r(dg, args.over, nwfs, args.verbose)
    if args.r2d: r2d(dg, args.over, nwfs, args.verbose)
    if args.r2d_file:
        f_raw, f_dsp = args.r2d_file
        r2d_file(f_raw, f_dsp, args.over, nwfs, args.verbose)


def load_datagroup():
    """
    """
    dg = DataGroup('oppi.json')
    dg.load_df('oppi_fileDB.h5')

    # various filters can go here

    # que = 'run==0'
    # que = 'cycle == 2027'
    # dg.file_keys.query(que, inplace=True)

    # dg.file_keys = dg.file_keys[:1]

    print('files to process:')
    print(dg.file_keys)

    return dg


def d2r(dg, overwrite=False, nwfs=None, vrb=False):
    """
    run daq_to_raw on the current DataGroup
    """
    # print(dg.file_keys)
    # print(dg.file_keys.columns)

    subs = dg.subsystems # can be blank: ['']
    # subs = ['geds'] # TODO: ignore other datastreams
    # chans = ['g035', 'g042'] # TODO: select a subset of detectors

    print(f'Processing {dg.file_keys.shape[0]} files ...')

    for i, row in dg.file_keys.iterrows():

        f_daq = f"{dg.daq_dir}/{row['daq_dir']}/{row['daq_file']}"
        f_raw = f"{dg.lh5_dir}/{row['raw_path']}/{row['raw_file']}"
        subrun = row['cycle'] if 'cycle' in row else None

        if not overwrite and os.path.exists(f_raw):
            print('file exists, overwrite not set, skipping f_raw:\n   ', f_raw)
            continue

        daq_to_raw(f_daq, f_raw, config=dg.config, subsystems=subs, verbose=vrb,
                   n_max=nwfs, overwrite=overwrite, subrun=subrun)#, chans=chans)


def r2d(dg, overwrite=False, nwfs=None, vrb=False):
    """
    """
    # print(dg.file_keys)
    # print(dg.file_keys.columns)

    with open(f'{dg.experiment}_dsp.json') as f:
        dsp_config = json.load(f, object_pairs_hook=OrderedDict)

    for i, row in dg.file_keys.iterrows():

        f_raw = f"{dg.lh5_dir}/{row['raw_path']}/{row['raw_file']}"
        f_dsp = f"{dg.lh5_dir}/{row['dsp_path']}/{row['dsp_file']}"

        if "sysn" in f_raw:
            tmp = {'sysn' : 'geds'} # hack for lpgta
            f_raw = f_raw.format_map(tmp)
            f_dsp = f_dsp.format_map(tmp)
            
        if not overwrite and os.path.exists(f_dsp):
            print('file exists, overwrite not set, skipping f_dsp:\n   ', f_dsp)
            continue

        raw_to_dsp(f_raw, f_dsp, dsp_config, n_max=nwfs, verbose=vrb,
                   overwrite=overwrite)


def r2d_file(f_raw, f_dsp, overwrite=True, nwfs=None, vrb=False):
    """
    single-file mode, for testing
    """
    print('raw_to_dsp, single-file mode.')
    print('  input:', f_raw)
    print('  output:', f_dsp)
    
    # always overwrite
    if os.path.exists(f_dsp):
        os.remove(f_dsp)
    
    with open('oppi_dsp.json') as f:
        dsp_config = json.load(f, object_pairs_hook=OrderedDict)
        
    raw_to_dsp(f_raw, f_dsp, dsp_config, n_max=nwfs, verbose=vrb, 
               overwrite=overwrite)
    


if __name__=="__main__":
    main()
