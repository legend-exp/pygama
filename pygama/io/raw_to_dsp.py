#! /usr/bin/env python3

import os
import sys
import json
import h5py
import time
import numpy as np
from collections import OrderedDict
from pprint import pprint
import argparse

from pygama import __version__ as pygama_version
from pygama.io import lh5
from pygama.utils import update_progress
import pygama.git as git
from pygama.dsp.build_processing_chain import *

def raw_to_dsp(f_raw, f_dsp, dsp_config, lh5_tables=None, database=None,
               outputs=None, n_max=np.inf, overwrite=True, buffer_len=3200, 
               block_width=16, verbose=1):
    """
    Uses the ProcessingChain class.
    The list of processors is specifed via a JSON file.
    """
    t_start = time.time()

    if isinstance(dsp_config, str):
        with open(dsp_config, 'r') as config_file:
            dsp_config = json.load(config_file, object_pairs_hook=OrderedDict)
    
    if not isinstance(dsp_config, dict):
        raise Exception('Error, dsp_config must be an dict')

    raw_store = lh5.Store()
    lh5_file = raw_store.gimme_file(f_raw, 'r')
    if lh5_file is None:
        print(f'raw_to_dsp: input file not found: {f_raw}')
        return
    else: print(f'Opened file {f_raw}')

    # if no group is specified, assume we want to decode every table in the file
    if lh5_tables is None:
        lh5_tables = []
        lh5_tables_temp = raw_store.ls(f_raw)

        # sometimes 'raw' is nested, e.g g024/raw
        for tb in lh5_tables_temp:
            if "raw" not in tb:
                tbname = raw_store.ls(lh5_file[tb])[0]
                if "raw" in tbname:
                    tb = tb +'/'+ tbname # g024 + /raw
            lh5_tables.append(tb)
    
    # make sure every group points to waveforms, if not, remove the group
    for tb in lh5_tables:
        if 'raw' not in tb:
            lh5_tables.remove(tb)

    # get the database parameters. For now, this will just be a dict in a json
    # file, but eventually we will want to interface with the metadata repo
    if isinstance(database, str):
        with open(database, 'r') as db_file:
            database = json.load(db_file)

    if database and not isinstance(database, dict):
        database = None
        print('database is not a valid json file or dict. Using default db values.')
    
    # delete the old file. TODO: ONCE BUGS ARE FIXED IN LH5 MODULE, DO THIS ONLY IF OVERWRITE IS TRUE!
    try:
        os.remove(f_dsp)
        print("Deleted", f_dsp)
    except:
        pass
    
    for tb in lh5_tables:
        # load primary table and build processing chain and output table
        tot_n_rows = raw_store.read_n_rows(tb, f_raw)
        if n_max and n_max<tot_n_rows: tot_n_rows=n_max

        chan_name = tb.split('/')[0]
        db_dict = database.get(chan_name) if database else None
        lh5_in, n_rows_read = raw_store.read_object(tb, f_raw, 0, buffer_len)
        pc, tb_out = build_processing_chain(lh5_in, dsp_config, db_dict, outputs, verbose, block_width)
        
        print(f'Processing table: {tb} ...')
        for start_row in range(0, tot_n_rows, buffer_len):
            if verbose > 0:
                update_progress(start_row/tot_n_rows)
            lh5_in, n_rows = raw_store.read_object(tb, f_raw, start_row=start_row, obj_buf=lh5_in)
            n_rows = min(tot_n_rows-start_row, n_rows)
            pc.execute(0, n_rows)
            raw_store.write_object(tb_out, tb.replace('/raw', '/dsp'), f_dsp, n_rows=n_rows)
            
        if verbose > 0:
            update_progress(1)
        print(f'Done.  Writing to file {f_dsp}')

    # write processing metadata
    dsp_info = lh5.Struct()
    dsp_info.add_field('timestamp', lh5.Scalar(np.uint64(time.time())))
    dsp_info.add_field('python_version', lh5.Scalar(sys.version))
    dsp_info.add_field('numpy_version', lh5.Scalar(np.version.version))
    dsp_info.add_field('h5py_version', lh5.Scalar(h5py.version.version))
    dsp_info.add_field('hdf5_version', lh5.Scalar(h5py.version.hdf5_version))
    dsp_info.add_field('pygama_version', lh5.Scalar(pygama_version))
    dsp_info.add_field('pygama_branch', lh5.Scalar(git.branch))
    dsp_info.add_field('pygama_revision', lh5.Scalar(git.revision))
    dsp_info.add_field('pygama_date', lh5.Scalar(git.commit_date))
    dsp_info.add_field('dsp_config', lh5.Scalar(json.dumps(dsp_config, indent=2)))
    raw_store.write_object(dsp_info, 'dsp_info', f_dsp)

    t_elap = (time.time() - t_start) / 60
    print(f'Done processing.  Time elapsed: {t_elap:.2f} min.')
