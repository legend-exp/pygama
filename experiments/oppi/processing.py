#!/usr/bin/env python3
import sys, os
import json
import numpy as np
import argparse
import pandas as pd
import subprocess as sp
from pprint import pprint
from collections import OrderedDict
import tinydb as db
from tinydb.storages import MemoryStorage

from pygama import DataGroup
import pygama.lh5 as lh5
from pygama.io.daq_to_raw import daq_to_raw
from pygama.io.raw_to_dsp import raw_to_dsp


def main():
    doc="""
    OPPI STC data processing routine.
    """
    rthf = argparse.RawTextHelpFormatter
    par = argparse.ArgumentParser(description=doc, formatter_class=rthf)
    arg, st, sf = par.add_argument, 'store_true', 'store_false'

    # declare datagroup
    arg('-q', '--query', nargs=1, type=str,
        help="select file group to calibrate: -q 'run==1' ")

    # routines
    arg('--d2r', action=st, help='run daq_to_raw')
    arg('--r2d', action=st, help='run raw_to_dsp')
    arg('--r2d_file', nargs=2, type=str, help='single-file raw_to_dsp')
    arg('--d2h', action=st, help='run dsp_to_hit (CAGE-specific)')

    # options
    arg('-o', '--over', action=st, help='overwrite existing files')
    arg('-n', '--nwfs', nargs='*', type=int, help='limit num. waveforms')
    arg('-v', '--verbose', action=st, help='verbose mode')

    args = par.parse_args()

    # load main DataGroup, select files to calibrate
    dg = DataGroup('oppi.json', load=True)
    if args.query:
        que = args.query[0]
        dg.fileDB.query(que, inplace=True)
    else:
        dg.fileDB = dg.fileDB[-1:]

    view_cols = ['run','cycle','daq_file','runtype','startTime']#,'threshold',
                 # 'stopTime','runtime']
    print(dg.fileDB[view_cols].to_string())
    print('Files:', len(dg.fileDB))
    # exit()

    # -- set options --
    nwfs = args.nwfs[0] if args.nwfs is not None else np.inf

    print('Processing settings:'
          # '\n$LPGTA_DATA =', os.environ.get('LPGTA_DATA'),
          # '\n$LEGEND_META =', os.environ.get('LEGEND_META'),
          f'\n  overwrite? {args.over}'
          f'\n  limit wfs? {nwfs}')

    # -- run routines --
    if args.d2r: d2r(dg, args.over, nwfs, args.verbose)
    if args.r2d: r2d(dg, args.over, nwfs, args.verbose)
    if args.d2h: d2h(dg, args.over, nwfs, args.verbose)

    if args.r2d_file:
        f_raw, f_dsp = args.r2d_file
        r2d_file(f_raw, f_dsp, args.over, nwfs, args.verbose)


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
        # f_raw = 'test.lh5'
        subrun = row['cycle'] if 'cycle' in row else None

        if not overwrite and os.path.exists(f_raw):
            print('file exists, overwrite not set, skipping f_raw:\n   ', f_raw)
            continue

        daq_to_raw(f_daq, f_raw, config=dg.config, systems=subs, verbose=vrb,
                   n_max=nwfs, overwrite=overwrite, subrun=subrun)#, chans=chans)


def r2d(dg, overwrite=False, nwfs=None, vrb=False):
    """
    """
    # print(dg.fileDB)
    # print(dg.fileDB.columns)

    with open(f'config_dsp.json') as f:
        dsp_config = json.load(f, object_pairs_hook=OrderedDict)

    for i, row in dg.fileDB.iterrows():

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


def d2h(dg, overwrite=False, nwfs=None, vrb=False):
    """
    """
    # merge main and ecal config JSON as dicts
    config = dg.config
    with open(config['ecal_config']) as f:
        config = {**dg.config, **json.load(f)}
    dg.config = config

    for i, row in dg.fileDB.iterrows():

        f_dsp = f"{dg.lh5_dir}/{row['dsp_path']}/{row['dsp_file']}"
        f_hit = f"{dg.lh5_dir}/{row['hit_path']}/{row['hit_file']}"

        if not overwrite and os.path.exists(f_hit):
            print('file exists, overwrite not set, skipping f_hit:\n   ', f_dsp)
            continue

        t_start = row['startTime']
        dsp_to_hit_cage(f_dsp, f_hit, dg, n_max=nwfs, verbose=vrb, t_start=t_start)


def dsp_to_hit_cage(f_dsp, f_hit, dg, n_max=None, verbose=False, t_start=None):
    """
    non-general placeholder for creating a pygama 'hit' file.  uses pandas.
    for every file, apply:
    - energy calibration (peakfit results)
    - timestamp correction
    for a more general dsp_to_hit, maybe each function could be given in terms
    of an 'apply' on a dsp dataframe ...
    
    TODO: create entry config['rawe'] with list of energy pars to calibrate, as 
    in energy_cal.py
    """
    rawe = ['trapEmax']
    
    # create initial 'hit' DataFrame from dsp data
    hit_store = lh5.Store()
    data = hit_store.read_object(dg.config['input_table'], f_dsp)
    df_hit = data.get_dataframe()
    
    # 1. get energy calibration for this run from peakfit 
    cal_db = db.TinyDB(storage=MemoryStorage)
    with open(dg.config['ecaldb']) as f:
        raw_db = json.load(f)
        cal_db.storage.write(raw_db)
    runs = dg.fileDB.run.unique()
    if len(runs) > 1:
        print("sorry, I can't do combined runs yet")
        exit()
    run = runs[0]
    for etype in rawe:
        tb = cal_db.table(f'peakfit_{etype}').all()
        df_cal = pd.DataFrame(tb)
        df_cal['run'] = df_cal['run'].astype(int)
        df_run = df_cal.loc[df_cal.run==run]
        cal_pars = df_run.iloc[0][['cal0','cal1','cal2']]
        pol = np.poly1d(cal_pars) # handy numpy polynomial object
        df_hit[f'{etype}_cal'] = pol(df_hit[f'{etype}'])

    # 2. compute timestamp rollover correction (specific to struck 3302)
    clock = 100e6 # 100 MHz
    UINT_MAX = 4294967295 # (0xffffffff)
    t_max = UINT_MAX / clock
    ts = df_hit['timestamp'].values / clock
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
    df_hit['ts_sec'] = np.concatenate(ts_new)
    
    # 3. compute global timestamp
    if t_start is not None:
        df_hit['ts_glo'] = df_hit['ts_sec'] + t_start 
    
    # write to LH5 file
    if os.path.exists(f_hit):
        os.remove(f_hit)
    sto = lh5.Store()
    tb_name = dg.config['input_table'].replace('dsp', 'hit')
    tb_lh5 = lh5.Table(size=len(df_hit))
    
    for col in df_hit.columns:
        tb_lh5.add_field(col, lh5.Array(df_hit[col].values, attrs={'units':''}))
        print(col)
    
    print(f'Writing table: {tb_name} in file:\n   {f_hit}')
    sto.write_object(tb_lh5, tb_name, f_hit)
    



if __name__=="__main__":
    main()
