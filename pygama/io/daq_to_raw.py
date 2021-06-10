"""
pygama tier 0 processing
raw daq data --> pandas dfs saved to hdf5 file (tier 1)
"""
import os, sys, time, json
import numpy as np
from parse import parse

from ..utils import *
from ..io.orcadaq import *
from ..io.llamadaq import *
from ..io.compassdaq import *
from ..io.fcdaq import *


def daq_to_raw(daq_filename, raw_file_pattern=None, subrun=None, systems=None,
               n_max=np.inf, verbose=False, out_dir=None, chans=None,
               overwrite=True, config={}):
    """
    Convert DAQ files into LEGEND-HDF5 `raw` format.
    Takes an input file (daq_filename) and an output file (raw_file_pattern).

    If the list `systems` is supplied, the raw_file_pattern should be a string
    containing `{sysn}`, which is used to create a list of output files for
    each data taker.

    Config options for daq_to_raw:
        buffer_size (int): default length to use for tables
        ch_groups (dict): associates groups of channels to be in the same or
            a different output LH5 table. See ch_group.py
    """
    # convert any environment variables
    daq_filename = os.path.expandvars(daq_filename)
    raw_file_pattern = os.path.expandvars(raw_file_pattern)

    # load options from config (can be dict or JSON filename)
    if isinstance(config, str):
        with open(os.path.expandvars(config)) as f:
            config = json.load(f)
    d2r_conf = config['daq_to_raw'] if 'daq_to_raw' in config else config
    buffer_size = d2r_conf['buffer_size'] if 'buffer_size' in d2r_conf else 8192

    # if we're not given a raw filename, make a simple one with subrun number
    if raw_file_pattern is None:
        if subrun is None:
            print('Error, must supply either raw_file_pattern or run number!')
            exit()
        if out_dir is None: out_dir = './'
        if systems is None:
            raw_file_pattern = f'{out_dir}/raw_run{subrun}.lh5'
        else:
            raw_file_pattern = f'{out_dir}/raw_{{sysn}}_subrun{subrun}.lh5'

    # set up write to multiple output files
    if isinstance(systems, list):
        raw_files = {sysn: raw_file_pattern.replace('{sysn}', sysn) for sysn in systems}
    else:
        raw_files = {'default': raw_file_pattern}

    # clear existing output files
    if overwrite:
        for sysn, file in raw_files.items():
            if os.path.isfile(file):
                if verbose:
                    print('Overwriting existing file:', file)
                os.remove(file)

    # if verbose:
    print('Starting daq_to_raw processing.'
          f'\n  Buffer size: {buffer_size}'
          f'\n  Max num. events: {n_max}'
          f'\n  Cycle (subrun) num: {subrun}'
          f'\n  Input: {daq_filename}\n  Output:')
    pprint(raw_files)

    t_start = time.time()
    bytes_processed = None


    ch_groups_dict = None
    if 'ch_groups' in d2r_conf: ch_groups_dict = d2r_conf['ch_groups']

    # get the DAQ mode
    if config['daq'] == 'ORCA':
        print('note, remove decoder input option')
        process_orca(daq_filename, raw_file_pattern, n_max, ch_groups_dict, verbose, buffer_size=buffer_size)

    elif config['daq'] == 'FlashCam':
        print("Processing FlashCam ...")
        bytes_processed = process_flashcam(daq_filename, raw_files, n_max, ch_groups_dict, verbose, buffer_size=buffer_size, chans=chans)

    elif config['daq'] == 'SIS3316':
        process_llama_3316(daq_filename, raw_file_pattern, run, n_max, config, verbose)

    elif config['daq'] == 'CAENDT57XXDecoder':
        print('note, remove decoder input option')
        process_compass(daq_filename, raw_file_pattern, None, out_dir)

    else:
        print(f"DAQ: {config['daq']} not recognized.  Exiting ...")
        exit()

    # --------- summary ------------

    elapsed = time.time() - t_start
    print("Time elapsed: {:.2f} sec".format(elapsed))
    if 'sysn' not in raw_file_pattern:
        statinfo = os.stat(raw_file_pattern)
        print('File size: {}'.format(sizeof_fmt(statinfo.st_size)))
        print('Conversion speed: {}ps'.format(sizeof_fmt(statinfo.st_size/elapsed)))
        print('  Output file:', raw_file_pattern)
    else:
        print('Total converted: {}'.format(sizeof_fmt(bytes_processed)))
        print('Conversion speed: {}ps'.format(sizeof_fmt(bytes_processed/elapsed)))

    print('Done.\n')
