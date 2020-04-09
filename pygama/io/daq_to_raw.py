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


def daq_to_raw(daq_filename, raw_filename=None, run=None, prefix="t1",
               suffix="lh5", chan_list=None, n_max=np.inf,
               verbose=False, output_dir=None, overwrite=True, decoders=None,
               config={}):
    """
    """
    print()
    print("Starting pygama daq_to_raw processing ...")

    daq_filename = os.path.expandvars(daq_filename)
    print("  Input file:", daq_filename)

    if isinstance(config, str):
        config = os.path.expandvars(config)
        with open(config) as f:
            config = json.load(f)

    d2r_conf = config['daq_to_raw'] if 'daq_to_raw' in config else config
    if 'daq_filename_template' in d2r_conf:
        filename_info = parse(d2r_conf['daq_filename_template'], daq_filename.split('/')[-1]).named
        if 'filename_info' not in d2r_conf: d2r_conf['filename_info'] = {}
        d2r_conf['filename_info'].update(filename_info)
        if 'filename_info_mods' in d2r_conf:
            for key, value in d2r_conf['filename_info_mods'].items():
                d2r_conf['filename_info'][key] = value[d2r_conf['filename_info'][key]]


    if raw_filename is None:
        if 'raw_filename_template' in d2r_conf:
            raw_filename = d2r_conf['raw_filename_template']
            class SafeDict(dict):
                def __missing__(self, key):
                    return '{' + key + '}'
            if run is not None: raw_filename = raw_filename.format_map(SafeDict({ 'run': int(run) }))
            raw_filename = raw_filename.format_map(SafeDict(d2r_conf['filename_info']))
        elif run is not None: raw_filename = f"{prefix}_run{run}.{suffix}"
        else:
            print('daq_to_raw error: must supply either raw_filename or run number')
            sys.exit()

    if output_dir is None:
        output_dir = config['raw_dir'] if 'raw_dir' in config else os.getcwd()
    output_dir = os.path.expandvars(output_dir)
    raw_filename = f"{output_dir}/{raw_filename}"
    print('  Output:', raw_filename)


    # ###############################################################
    # # Change for HADES style output
    # # TODO: as vince would say, i need to do something smart here
    #
    # if ftype == "hades_char":
    # # declare Tier 1 output file
    #
    #    file_body = daq_filename.split("/")[-1].replace("fcio","h5")
    #    raw_filename = "{}/{}_{}".format(output_dir,prefix,file_body)
    #    if os.path.isfile(raw_filename):
    #        if overwrite:
    #            print("Overwriting existing file...")
    #            os.remove(raw_filename)
    #        else:
    #            print("File already exists, continuing ...")
    #            return
    # ################################################################
    # else:

    if os.path.isfile(raw_filename):
       if overwrite:
           print("Overwriting existing file...")
           os.remove(raw_filename)
       else:
           print("File already exists, continuing ...")
           return

    if 'file_label' in raw_filename:
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

    buffer_size = d2r_conf['buffer_size'] if 'buffer_size' in d2r_conf else 8096

    t_start = time.time()

    # set max number of events (useful for debugging)
    if n_max is not np.inf and n_max is not None:
        n_max = int(n_max)

    bytes_processed = 0
    # get the DAQ mode
    if config["daq"] == "ORCA":
        process_orca(daq_filename, raw_filename, n_max, decoders, config, verbose, run=run, buffer_size=buffer_size)

    elif config["daq"] == "FlashCam":
        bytes_processed = process_flashcam(daq_filename, raw_filename, n_max, config, verbose, buffer_size=buffer_size)

    elif config["daq"] == "SIS3316":
        process_llama_3316(daq_filename, raw_filename, run, n_max, config, verbose)

    elif config["daq"] == "CAENDT57XXDecoder":
        process_compass(daq_filename, raw_filename, decoders, output_dir)

    else:
        print(f"DAQ: {config['daq']} not recognized.  Exiting ...")
        exit()

    # --------- summary ------------

    elapsed = time.time() - t_start
    print("Time elapsed: {:.2f} sec".format(elapsed))
    if 'file_label' not in raw_filename:
        statinfo = os.stat(raw_filename)
        print("File size: {}".format(sizeof_fmt(statinfo.st_size)))
        print("Conversion speed: {}ps".format(sizeof_fmt(statinfo.st_size/elapsed)))
        print("  Output file:", raw_filename)
    else:
        print("Total converted: {}".format(sizeof_fmt(bytes_processed)))
        print("Conversion speed: {}ps".format(sizeof_fmt(bytes_processed/elapsed)))
    print("Done.\n")


