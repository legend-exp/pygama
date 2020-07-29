#!/usr/bin/env python3
import os
import json
import argparse
import pandas as pd
import numpy as np
from pprint import pprint
from datetime import datetime
import tinydb as db
from tinydb.storages import MemoryStorage

import matplotlib
if 'cenpa-rocks' in os.environ.get('HOSTNAME'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('../../pygama/clint.mpl')

from pygama import DataGroup
from pygama.io.orcadaq import parse_header
import pygama.io.lh5 as lh5
import pygama.analysis.metadata as pmd
import pygama.analysis.histograms as pgh

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from tqdm import tqdm
    tqdm.pandas() # suppress annoying FutureWarning


def main():
    doc="""
    === pygama: ecal_db.py =====================================================
    
    energy calibration app

    - Uses a DataGroup to organize files and processing.
    - Metadata is handled in JSON format with 'legend-metadata' conventions.

    === T. Mathew, C. Wiseman (UW) =============================================
    """
    rthf = argparse.RawTextHelpFormatter
    par = argparse.ArgumentParser(description=doc, formatter_class=rthf)
    arg, st, sf = par.add_argument, 'store_true', 'store_false'
    arg('--init', action=st, help='initialize primary ecal file')
    arg('--spec', action=st, help='check uncalibrated energy histogram')
    args = par.parse_args()
    
    # load main DataGroup
    dg = DataGroup('oppi.json', load=True)
    file_db = dg.file_keys
    view_cols = ['run', 'cycle', 'startTime', 'runtime']

    # temporary: select files to calibrate
    que = 'run == 1'
    # que = 'rtype==calib'
    # que = 'cycle == 2180'
    file_db.query(que, inplace=True)
    # file_db = file_db[:1]
    # print(file_db[view_cols])
    # exit()

    # load the ecal db s/t the pretty on-disk formatting isn't changed
    f_db = dg.config['ecaldb']
    ecal_db = db.TinyDB(storage=MemoryStorage)
    with open(f_db) as f:
        raw_db = json.load(f)
        ecal_db.storage.write(raw_db)

    # -- run routines -- 
    if args.init: 
        setup_ecaldb(dg) # first time only!
    if args.spec: 
        check_raw_spectrum(dg, ecal_db)
        exit()
    
    # select options. can either group by runDB or calibrate files individually
    runDB = True
    etypes = ['energy', 'trapEmax'] # energy, trapEmax
    
    # determine these from check_raw_spectrum (xlo, xhi, xpb).  maybe add to ecaldb
    ebins = {'energy':[0, 1e6, 1000], 'trapEmax':[0, 10000, 10]}
    
    gb_cols = ['run'] if runDB else ['cycle']
    gb_args = {'dg':dg, 'etypes':etypes, 'ecal_db':ecal_db}
    
    # run calibration 
    result = file_db.groupby(gb_cols).apply(calibrate_group, **gb_args)
    # result = file_db.groupby(gb_cols).progress_apply(calibrate_group, **gb_args)
    # print(result)


def check_raw_spectrum(dg, ecal_db=None):
    """
    $ ./ecal_db.py --spec
    use this to find a good binning for the initial uncalibrated spectrum
    """
    etype, xlo, xhi, xpb = 'energy', 0, 1e6, 1000
    etype, xlo, xhi, xpb = 'trapEmax', 0, 10000, 10

    # browse first file in DataGroup and show columns of input table
    df_row = dg.file_keys.iloc[0]
    f_dsp = dg.lh5_dir + df_row['dsp_path'] + '/' + df_row['dsp_file']
    sto = lh5.Store()
    
    # load energy data for this estimator
    sto = lh5.Store()
    query = db.Query()
    file_info = ecal_db.table('_file_info').get(query)
    tb_in = file_info['input_table']
    data = sto.read_object(f'{tb_in}/{etype}', f_dsp).nda
    print(etype, len(data))
    
    # generate histogram
    hist, bins, var = pgh.get_hist(data, range=(xlo, xhi), dx=xpb)
    bins = bins[1:] # trim zero bin, not needed with ds='steps'

    plt.plot(bins, hist, ds='steps', c='b', lw=2, label=etype)
    plt.xlabel(etype, ha='right', x=1)
    plt.ylabel('Counts', ha='right', y=1)
    plt.savefig('./plots/cal_spec_test.png')
    exit()


def setup_ecaldb(dg):
    """
    one-time set up of primary database file
    """
    ans = input('(Re)create main ecal JSON file?  Are you really sure? (y/n) ')
    if ans.lower() != 'y':
        exit()
    
    f_db = dg.config['ecaldb'] # for pgt, should have one for each detector 
    
    if os.path.exists(f_db):
        os.remove(f_db)
    
    # create the database in-memory
    ecal_db = db.TinyDB(storage=MemoryStorage)
    query = db.Query()
    
    # create a table with metadata (provenance) about this calibration file
    file_info = {
        "system" : "oppi",
        "cal_type" : "energy",
        "created_gmt" : datetime.utcnow().strftime("%m/%d/%Y, %H:%M:%S"), 
        "input_table" : "/ORSIS3302DecoderForEnergy/raw"
        }
    ecal_db.table('_file_info').insert(file_info)
    # tb_info.insert(file_info)
        
    # pretty-print the JSON database to file
    raw_db = ecal_db.storage.read()
    pmd.write_pretty(raw_db, f_db)
    
    # show the file as-is on disk
    with open(f_db) as f:
        print(f.read()) 

    
def calibrate_group(df_run, dg=None, ecal_db=None, etypes=None):
    """
    """
    # get list of files to access
    dsp_list = dg.lh5_dir + df_run['dsp_path'] + '/' + df_run['dsp_file']
    # print(dsp_list)
    # print([f.split('/')[-1] for f in dsp_list])
    
    # determine hdf5 group in file for this detector
    query = db.Query()
    file_info = ecal_db.table('_file_info').get(query)
    tb_in = file_info['input_table']
    
    # load energy data for each estimator
    sto = lh5.Store()
    edata = {et : [] for et in etypes}
    for f_dsp in dsp_list:
        for et in etypes:
            data = sto.read_object(f'{tb_in}/{et}', f_dsp)
            edata[et].append(data.nda)
    edata = {et : np.concatenate(edata[et]) for et in etypes}
    print('Found energy data:', [(et, len(ev)) for et, ev in edata.items()])
    
    # get histogram and runtime
    
    
    exit()
    
    
if __name__=='__main__':
    main()