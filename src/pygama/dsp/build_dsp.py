#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time

import h5py
import numpy as np

import pygama
import pygama.lgdo.lh5_store as lh5
from pygama.dsp.errors import DSPFatal
from pygama.dsp.processing_chain import build_processing_chain
from pygama.math.utils import tqdm_range


def build_dsp(f_raw, f_dsp, dsp_config, lh5_tables=None, database=None,
               outputs=None, n_max=np.inf, write_mode='r', buffer_len=3200,
               block_width=16, verbose=1, chan_config=None):
    """
    Convert raw-tier LH5 data into dsp-tier LH5 data by running a sequence of
    processors via the ProcessingChain.

    Parameters
    ----------
    f_raw : str
        Name of raw LH5 file to read from
    f_dsp : str
        Name of dsp LH5 file to write to
    dsp_config : str
        Name of json file containing ProcessingChain config. See
        pygama.processing_chain.build_processing_chain for details
    lh5_tables
        DOC_ME
    database : str (optional)
        Name of json file containing a parameter database. See
        pygama.processing_chain.build_processing_chain for details
    outputs : list of strs (optional)
        List of parameter names to write to f_dsp. If not provided, use
        list provided in dsp_config
    n_max : int (optional)
        Number of waveforms to process. Default all.
    write_mode : 'r' 'a' or 'u' (optional)
        'r': Delete old file at f_dsp before writing
        'a': Append to end of existing f_dsp
        'u': Update values in existing f_dsp
    buffer_len : int (optional)
        Number of waveforms to read/write from disk at a time. Default 3200.
    block_width : int (optional)
        Number of waveforms to process at a time. Default 16.
    verbose : int (optional)
        0: Silent except for exceptions thrown
        1: Minimal information printed (default)
        2: Verbose output for debugging purposes
    chan_config
        DOC_ME
    """
    t_start = time.time()

    if isinstance(dsp_config, str):
        with open(dsp_config) as config_file:
            dsp_config = json.load(config_file)

    if not isinstance(dsp_config, dict):
        raise Exception('Error, dsp_config must be an dict')

    raw_store = lh5.LH5Store()
    lh5_file = raw_store.gimme_file(f_raw, 'r')
    if lh5_file is None:
        print(f'build_dsp: input file not found: {f_raw}')
        return
    else: print(f'Opened file {f_raw}')

    # if no group is specified, assume we want to decode every table in the file
    if lh5_tables is None:
        lh5_tables = []
        lh5_keys = lh5.ls(f_raw)

        # sometimes 'raw' is nested, e.g g024/raw
        for tb in lh5_keys:
            if "raw" not in tb:
                tbname = lh5.ls(lh5_file[tb])[0]
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

    # load DSP config (default: one config file for all tables)
    if isinstance(dsp_config, str):
        with open(dsp_config) as config_file:
            dsp_config = json.load(config_file, object_pairs_hook=OrderedDict)

    # get the database parameters. For now, this will just be a dict in a json
    # file, but eventually we will want to interface with the metadata repo
    if isinstance(database, str):
        with open(database) as db_file:
            database = json.load(db_file)

    if database and not isinstance(database, dict):
        database = None
        print('database is not a valid json file or dict. Using default db values.')

    # clear existing output files
    if write_mode=='r':
        if os.path.isfile(f_dsp):
            if verbose:
                print('Overwriting existing file:', f_dsp)
            os.remove(f_dsp)

    # write processing metadata
    dsp_info = lh5.Struct()
    dsp_info.add_field('timestamp', lh5.Scalar(np.uint64(time.time())))
    dsp_info.add_field('python_version', lh5.Scalar(sys.version))
    dsp_info.add_field('numpy_version', lh5.Scalar(np.version.version))
    dsp_info.add_field('h5py_version', lh5.Scalar(h5py.version.version))
    dsp_info.add_field('hdf5_version', lh5.Scalar(h5py.version.hdf5_version))
    dsp_info.add_field('pygama_version', lh5.Scalar(pygama.__version__))

    # loop over tables to run DSP on
    for tb in lh5_tables:
        # load primary table and build processing chain and output table
        tot_n_rows = raw_store.read_n_rows(tb, f_raw)
        if n_max and n_max<tot_n_rows: tot_n_rows=n_max

        # if we have separate DSP files for each table, read them in here
        if chan_config is not None:
            f_config = chan_config[tb]
            with open(f_config) as config_file:
                dsp_config = json.load(config_file, object_pairs_hook=OrderedDict)
            print('Processing table:', tb, 'with DSP config file:\n  ', f_config)

        if not isinstance(dsp_config, dict):
            raise Exception('Error, dsp_config must be an dict')

        chan_name = tb.split('/')[0]
        db_dict = database.get(chan_name) if database else None
        tb_name = tb.replace('/raw', '/dsp')

        write_offset = 0
        raw_store.gimme_file(f_dsp, 'a')
        if write_mode=='a' and lh5.ls(f_dsp, tb_name):
            write_offset = raw_store.read_n_rows(tb_name, f_dsp)

        # Main processing loop
        print(f'Processing table: {tb} ...')
        lh5_it = lh5.LH5Iterator(f_raw, tb, buffer_len = buffer_len)
        proc_chain = None
        for lh5_in, start_row, n_rows in lh5_it:
            # Initialize
            if proc_chain is None:
                proc_chain, lh5_it.field_mask, tb_out = build_processing_chain(lh5_in, dsp_config, db_dict, outputs, verbose, block_width)
                tqdm_it = iter(tqdm_range(0, tot_n_rows, buffer_len, verbose))

            n_rows = min(tot_n_rows-start_row, n_rows)
            try:
                proc_chain.execute(0, n_rows)
            except DSPFatal as e:
                # Update the wf_range to reflect the file position
                e.wf_range = f"{e.wf_range[0]+start_row}-{e.wf_range[1]+start_row}"
                raise e

            raw_store.write_object(obj=tb_out,
                                   name=tb_name,
                                   lh5_file=f_dsp,
                                   n_rows=n_rows,
                                   wo_mode= 'o' if write_mode=='u' else 'a',
                                   write_start=write_offset+start_row,
            )
            next(tqdm_it)

            if start_row+n_rows > tot_n_rows:
                break

        print(f'Done.  Writing to file {f_dsp}')

    raw_store.write_object(dsp_info, 'dsp_info', f_dsp)

    t_elap = (time.time() - t_start) / 60
    print(f'Done processing.  Time elapsed: {t_elap:.2f} min.')



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="""Process a single tier 1 LH5
    file and produce a tier 2 LH5 file using a json config file and
    build_dsp.""")

    arg = parser.add_argument
    arg('file', help="Input raw LH5 file.")
    arg('-g', '--group', default=None, action='append', type=str,
        help="Name of group in LH5 file. By default process all base groups. Supports wildcards.")
    arg('-j', '--jsonconfig', default=str(default_config), type=str,
        help="Name of json file used by build_dsp to construct the processing routines used. By default use the installed dsp_config.json.")
    arg('-d', '--dbfile', default=None, type=str,
        help="JSON file to read DB parameters from. Should be nested dict with channel at the top level, and parameters below that.")
    arg('-o', '--output',
        help="Name of output file. By default, output to ./t2_[input file name].")
    arg('-p', '--outpar', default=None, action='append', type=str,
        help="Add outpar to list of parameters written to file. By default use the list defined in outputs list in config json file.")

    arg('-n', '--nevents', default=None, type=int,
        help="Number of waveforms to process. By default do the whole file")

    arg('-v', '--verbose', action='store_const', const=2, default=1,
        help="Verbose output, useful for debugging")
    arg('-q', '--quiet', action='store_const', const=0, dest='verbose',
        help="Silent output, print only exceptions thrown")

    arg('-b', '--block', default=16, type=int,
        help="Number of waveforms to process simultaneously. Default is 8")
    arg('-c', '--chunk', default=3200, type=int,
        help="Number of waveforms to read from disk at a time. Default is 256. THIS IS NOT IMPLEMENTED YET!")

    arg('-r', '--recreate', action='store_const', const='r', dest='writemode', default='r',
        help="Overwrite file if it already exists. Default option. Multually exclusive with --update and --append")
    arg('-u', '--update', action='store_const', const='u', dest='writemode',
        help="Update existing file with new values. Useful with the --outpar option. Mutually exclusive with --recreate and --append THIS IS NOT IMPLEMENTED YET!")
    arg('-a', '--append', action='store_const', const='a', dest='writemode',
        help="Append values to existing file. Mutually exclusive with --recreate and --update THIS IS NOT IMPLEMENTED YET!")
    args = parser.parse_args()

    out = args.output
    if out is None:
        out = 't2_'+args.file[args.file.rfind('/')+1:].replace('t1_', '')

    build_dsp(args.file, out, args.jsonconfig, lh5_tables=args.group, database=args.dbfile, verbose=args.verbose, outputs=args.outpar, n_max=args.nevents, write_mode=args.writemode, buffer_len=args.chunk, block_width=args.block)
