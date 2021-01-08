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
from pygama.dsp.ProcessingChain import ProcessingChain
from pygama.dsp.units import *
from pygama import lh5
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
        lh5_keys = raw_store.ls(f_raw)

        # sometimes 'raw' is nested, e.g g024/raw
        for tb in lh5_keys:
            if "raw" not in tb:
                tbname = raw_store.ls(lh5_file[tb])[0]
                if "raw" in tbname:
                    tb = tb +'/'+ tbname # g024 + /raw
            lh5_tables.append(tb)
    
    # make sure every group points to waveforms, if not, remove the group
    for tb in lh5_tables:
        if 'raw' not in tb:
            lh5_tables.remove(tb)
    if len(lh5_tables) == 0:
        print("Empty lh5_tables, exiting...")
        sys.exit(1)

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



if __name__=="__main__":
    parser = argparse.ArgumentParser(description=
"""Process a single tier 1 LH5 file and produce a tier 2 LH5 file using a
json config file and raw_to_dsp.""")
    
    arg = parser.add_argument
    arg('file', help="Input (tier 1) LH5 file.")
    arg('-o', '--output',
        help="Name of output file. By default, output to ./t2_[input file name].")
    
    arg('-v', '--verbose', default=1, type=int,
        help="Verbosity level: 0=silent, 1=basic warnings, 2=verbose output, 3=debug. Default is 2.")
    
    arg('-b', '--block', default=16, type=int,
        help="Number of waveforms to process simultaneously. Default is 8")
    
    arg('-c', '--chunk', default=3200, type=int,
        help="Number of waveforms to read from disk at a time. Default is 256. THIS IS NOT IMPLEMENTED YET!")
    arg('-n', '--nevents', default=None, type=int,
        help="Number of waveforms to process. By default do the whole file")
    arg('-g', '--group', default=None, action='append', type=str,
        help="Name of group in LH5 file. By default process all base groups. Supports wildcards.")
    defaultconfig = os.path.dirname(os.path.realpath(__loader__.get_filename())) + '/dsp_config.json'
    arg('-j', '--jsonconfig', default=defaultconfig, type=str,
        help="Name of json file used by raw_to_dsp to construct the processing routines used. By default use dsp_config in pygama/apps.")
    arg('-p', '--outpar', default=None, action='append', type=str,
        help="Add outpar to list of parameters written to file. By default use the list defined in outputs list in config json file.")
    arg('-d', '--dbfile', default=None, type=str,
        help="JSON file to read DB parameters from. Should be nested dict with channel at the top level, and parameters below that.")
    arg('-r', '--recreate', action='store_const', const=0, dest='writemode',
        help="Overwrite file if it already exists. Default option. Multually exclusive with --update and --append")
    arg('-u', '--update', action='store_const', const=1, dest='writemode',
        help="Update existing file with new values. Useful with the --outpar option. Mutually exclusive with --recreate and --append THIS IS NOT IMPLEMENTED YET!")
    arg('-a', '--append', action='store_const', const=1, dest='writemode',
        help="Append values to existing file. Mutually exclusive with --recreate and --update THIS IS NOT IMPLEMENTED YET!")
    args = parser.parse_args()

    out = args.output
    if out is None:
        out = 't2_'+args.file[args.file.rfind('/')+1:].replace('t1_', '')

    raw_to_dsp(args.file, out, args.jsonconfig, lh5_tables=args.group, database=args.dbfile, verbose=args.verbose, outputs=args.outpar, n_max=args.nevents, overwrite=args.writemode==0, buffer_len=args.chunk, block_width=args.block)
