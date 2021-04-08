import os, time, json
import numpy as np

from ..utils import *
from orca.stream_orca import *
from stream_llama import *
from stream_compass import *
from stream_fc import *


def build_raw(in_filename, stream_type, raw_file_pattern=None, pattern_dict=None, 
              ch_groups=None, n_max=np.inf, verbose=False, overwrite=True, buffer_size=8192)
    """
    Convert data into LEGEND hdf5 `raw` format.  
    Takes an input file (in_filename) and an output file (raw_file_pattern).


    Parameters
    ----------
    in_filename : str or f 
        The name of the input file to be converted, including path. Can use
        environment variables.
    stream_type : str
        Options are "ORCA", "FlashCams", "Llama", "Compass", "MGDO"
    raw_file_pattern : str or f string (optional)
        The name or a pattern for names of the output file(s), including path.
        Can use environment variables.
        
    rfp_subs : dict (optional)
    ch_groups : dict (optional)
    n_max : int (optional)
    verbose : bool (optional)
    overwrite : bool (optional)
    buffer_size : int (optional)
    
    Config options for build_raw:
        buffer_size (int): default length to use for tables
        ch_groups (dict): associates groups of channels to be in the same or
            a different output LH5 table. See ch_group.py
    """
    # convert any environment variables
    in_filename = os.path.expandvars(in_filename)
    raw_file_pattern = os.path.expandvars(raw_file_pattern)

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
    print('Starting build_raw processing.'
          f'\n  Buffer size: {buffer_size}'
          f'\n  Max num. events: {n_max}'
          f'\n  Cycle (subrun) num: {subrun}'
          f'\n  Input: {in_filename}\n  Output:')
    pprint(raw_files)

    t_start = time.time()
    bytes_processed = None

    # get the DAQ mode
    if config['daq'] == 'ORCA':
        print('note, remove decoder input option')
        process_orca(in_filename, raw_file_pattern, n_max, ch_groups, verbose, buffer_size=buffer_size)

    elif config['daq'] == 'FlashCam':
        print("Processing FlashCam ...")
        bytes_processed = process_flashcam(in_filename, raw_files, n_max, ch_groups, verbose, buffer_size=buffer_size, chans=chans)

    elif config['daq'] == 'SIS3316':
        process_llama_3316(in_filename, raw_file_pattern, run, n_max, config, verbose)

    elif config['daq'] == 'CAENDT57XXDecoder':
        print('note, remove decoder input option')
        process_compass(in_filename, raw_file_pattern, None, out_dir)

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
