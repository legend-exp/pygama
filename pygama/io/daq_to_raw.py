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
               n_max=np.inf, verbose=False, output_dir=None,
               overwrite=True, config={}):
    """
    """
    daq_filename = os.path.expandvars(daq_filename)
    
    # if config is a JSON file, load it
    if isinstance(config, str):
        config = os.path.expandvars(config)
        with open(config) as f:
            config = json.load(f)
            
    # load daq_to_raw settings 
    d2r_conf = config['daq_to_raw'] if 'daq_to_raw' in config else config
    buffer_size = d2r_conf['buffer_size'] if 'buffer_size' in d2r_conf else 8096

    

    pprint(config)
    exit()
        
    # if we're not given a raw filename, try to infer one
    if raw_filename is None:
    
        # load filename attributes into a dict
        if 'daq_filename_template' in d2r_conf:
            f_temp = d2r_conf['daq_filename_template']
            f_name = daq_filename.split('/')[-1]
            f_info = parse(f_temp, f_name).named
            if 'file_info' not in d2r_conf:
                d2r_conf['file_info'] = {}
            d2r_conf['file_info'].update(f_info)
            
        if 'filename_info_mods' in d2r_conf:
            for key, value in d2r_conf['filename_info_mods'].items():
                d2r_conf['file_info'][key] = value[d2r_conf['file_info'][key]]

        if 'raw_filename_template' in d2r_conf:
            raw_filename = d2r_conf['raw_filename_template']
            
            if subrun is not None: 
                sd = SafeDict({'run': int(subrun)})
                raw_filename = raw_filename.format_map(sd)
            
            raw_filename = raw_filename.format_map(SafeDict(d2r_conf['file_info']))
        
        elif subrun is not None: 
            raw_filename = f"raw_run{subrun}.{suffix}"
        
        else:
            print('Error: must supply either raw_filename or run number')
            exit()

    # exit()

    # need to be able to delete all subsystem files [gNNN, spms, etc..]
    if os.path.isfile(raw_filename):
       if overwrite:
           print("Overwriting existing file...")
           os.remove(raw_filename)
       else:
           print("File already exists, continuing ...")
           return

    if 'sysn' in raw_filename:
       ch_groups = config['daq_to_raw']['ch_groups']
       for group, attrs in ch_groups.items():
           out_filename = raw_filename.format_map(attrs)
           if os.path.isfile(out_filename):
               if overwrite:
                   print("Overwriting existing file", out_filename)
                   os.remove(out_filename)
               else:
                   print("File", out_filename, "already exists, continuing ...")
                   return

               
    

    if verbose:
        print('Starting daq_to_raw processing.'
              f'\n  Input: {daq_filename}'
              f'\n  Output: {raw_filename}'
              f'\n  Buffer size: {buffer_size}')

    exit()
    
    t_start = time.time()

    # set max number of events (useful for debugging)
    if n_max is not np.inf and n_max is not None:
        n_max = int(n_max)

    bytes_processed = 0
    # get the DAQ mode
    if config["daq"] == "ORCA":
        print('note, remove decoder input option')
        process_orca(daq_filename, raw_filename, n_max, None, config, verbose, run=run, buffer_size=buffer_size)

    elif config["daq"] == "FlashCam":
        bytes_processed = process_flashcam(daq_filename, raw_filename, n_max, config, verbose, buffer_size=buffer_size)

    elif config["daq"] == "SIS3316":
        process_llama_3316(daq_filename, raw_filename, run, n_max, config, verbose)

    elif config["daq"] == "CAENDT57XXDecoder":
        print('note, remove decoder input option')
        process_compass(daq_filename, raw_filename, None, output_dir)

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
