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


def daq_to_raw(daq_filename, raw_filename=None, subrun=None, subsystems=None, 
               n_max=np.inf, verbose=False, out_dir=None, chans=None,
               overwrite=True, config={}):
    """
    Convert DAQ files into LEGEND-HDF5 `raw` format.  
    Takes an input file (daq_filename) and an output file (raw_filename).
    
    If the list `subsystems` is supplied, the raw_filename should be a string
    containing `{sysn}`, which is used to create a list of output files for
    each data taker.
    """
    # convert any environment variables
    daq_filename = os.path.expandvars(daq_filename)
    raw_filename = os.path.expandvars(raw_filename)
    
    # load options from config (can be dict or JSON filename)
    if isinstance(config, str):
        with open(os.path.expandvars(config)) as f:
            config = json.load(f)
    d2r_conf = config['daq_to_raw'] if 'daq_to_raw' in config else config
    buffer_size = d2r_conf['buffer_size'] if 'buffer_size' in d2r_conf else 8096

    # if we're not given a raw filename, make a simple one with subrun number
    if raw_filename is None:
        if subrun is None:
            print('Error, must supply either raw_filename or run number!')
            exit()
        if out_dir is None: out_dir = './'
        if subsystems is None:
            raw_filename = f'{out_dir}/raw_run{subrun}.lh5'
        else:
            raw_filename = f'{out_dir}/raw_{{sysn}}_subrun{subrun}.lh5'

    # set up write to multiple output files
    if isinstance(subsystems, list):
        raw_files = {sysn: raw_filename.replace('{sysn}', sysn) for sysn in subsystems}
    else:
        raw_files = {'default': raw_filename}
        
    # clear existing output files
    if overwrite:
        for sysn, file in raw_files.items():
            if os.path.isfile(file):
                if verbose:
                    print('Overwriting existing file :', file)
                os.remove(file)
    
    # if verbose:
    print('Starting daq_to_raw processing.'
          f'\n  Buffer size: {buffer_size}'
          f'\n  Max num. events: {n_max}'
          f'\n  Input: {daq_filename}\n  Output:')
    pprint(raw_files)
    
    t_start = time.time()
    bytes_processed = None
    
    # get the DAQ mode
    if config['daq'] == 'ORCA':
        print('note, remove decoder input option')
        process_orca(daq_filename, raw_filename, n_max, None, config, verbose, run=run, buffer_size=buffer_size)

    elif config['daq'] == 'FlashCam':
        bytes_processed = process_flashcam(daq_filename, raw_files, n_max, config, verbose, buffer_size=buffer_size, chans=chans)

    elif config['daq'] == 'SIS3316':
        process_llama_3316(daq_filename, raw_filename, run, n_max, config, verbose)

    elif config['daq'] == 'CAENDT57XXDecoder':
        print('note, remove decoder input option')
        process_compass(daq_filename, raw_filename, None, out_dir)

    else:
        print(f"DAQ: {config['daq']} not recognized.  Exiting ...")
        exit()

    # --------- summary ------------

    elapsed = time.time() - t_start
    print("Time elapsed: {:.2f} sec".format(elapsed))
    if 'sysn' not in raw_filename:
        statinfo = os.stat(raw_filename)
        print('File size: {}'.format(sizeof_fmt(statinfo.st_size)))
        print('Conversion speed: {}ps'.format(sizeof_fmt(statinfo.st_size/elapsed)))
        print('  Output file:', raw_filename)
    else:
        print('Total converted: {}'.format(sizeof_fmt(bytes_processed)))
        print('Conversion speed: {}ps'.format(sizeof_fmt(bytes_processed/elapsed)))
    
    print('Done.\n')
