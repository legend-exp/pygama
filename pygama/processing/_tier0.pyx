""" ========= PYGAMA =========
TIER 0 MAIN PROCESSING ROUTINE
Cythonized for maximum speed.
"""
cimport numpy as np

import numpy as np
import os, re, sys, glob
import pandas as pd
import h5py
from future.utils import iteritems
from multiprocessing import Pool, cpu_count
from functools import partial

from ..utils import update_progress
from ..decoders.digitizers import *
from ..decoders.dataloading import *
from .header_parser import *
from .base_classes import *


def ProcessTier0(filename,
                 output_file_string="t1",
                 chan_list=None,
                 n_max=np.inf,
                 verbose=False,
                 output_dir=None,
                 decoders=None):
    """ Reads in "raw / tier 0" ORCA data and saves to an hdf5 format using pandas
    filename: path to an orca data file
    output_file_string: output file name will be <output_file_string>_run<runNumber>.h5
    n_max: maximum number of events to process (useful for debugging)
    verbose: spits out a progressbar to let you know how the processing is going
    output_dir: where to stash the t1 file
    """
    print("Starting pygama Tier 0 processing ...")
    print("  Input file: "+filename)

    SEEK_END = 2

    directory = os.path.dirname(filename)
    output_dir = os.getcwd() if output_dir is None else output_dir

    # parse the header
    reclen, reclen2, header_dict = parse_header(filename)
    print("Header parsed.")
    print("   %d longs (in plist header)" % reclen)
    print("   %d bytes in the header" % reclen2)

    f_in = open(filename.encode('utf-8'), "rb")
    if f_in == None:
        print("Couldn't find the file %s" % filename)
        sys.exit(0)

    #figure out the total size
    f_in.seek(0, SEEK_END)
    file_size = float(f_in.tell())
    f_in.seek(0, 0)  # rewind
    file_size_MB = file_size / 1e6
    print("Total file size: %3.3f MB" % file_size_MB)

    # skip the header
    # reclen is in number of longs, and we want to skip a number of bytes
    f_in.seek(reclen * 4)

    # pull out the run number
    runNumber = get_run_number(header_dict)
    print("Run number: {}".format(runNumber))

    # pull out the data IDs
    id_dict = get_decoder_for_id(header_dict)
    print("The Data IDs present in this file (header) are:")
    for id in id_dict:
        print("    {}: {}".format(id, id_dict[id]))
    used_decoder_names = set([id_dict[id] for id in id_dict])

    # get pygama's available decoders
    print("Available pygama decoders:")
    if decoders is None:
        decoders = get_decoders(header_dict)
        decoder_names = [d.decoder_name for d in decoders]
    for d in decoder_names:
        print("    -- {}".format(d))

    # kill unnecessary decoders
    final_decoder_list = list(
        set(decoder_names).intersection(used_decoder_names))
    decoders = [d for d in decoders if d.decoder_name in final_decoder_list]
    decoder_to_id = {d.decoder_name: d for d in decoders}

    print("Applying these decoders to the file:")
    for name in final_decoder_list:
        for id in id_dict:
            if id_dict[id] == name:
                this_data_id = id
        print("    {}: {}".format(this_data_id, name))

    # keep track of warnings we've raised for missing decoders
    unrecognized_data_ids = []
    board_id_map = {}
    appended_data_map = {}

    print("Beginning Tier 0 processing of file:\n    {}...".format(filename))
    event_number = 0  #number of events decoded
    while (event_number < n_max and f_in.tell() < file_size):
        event_number += 1
        if verbose and event_number % 1000 == 0:
            update_progress(float(f_in.tell()) / file_size)

        try:
            event_data, data_id = get_next_event(f_in)
        except EOFError:
            break
        except Exception as e:
            print("Failed to get the next event... (Exception: {})".format(e))
            break

        try:
            decoder = decoder_to_id[id_dict[data_id]]
        except KeyError:
            if data_id not in id_dict and data_id not in unrecognized_data_ids:
                unrecognized_data_ids.append(data_id)
            continue

        # sends data to the pandas dataframe
        decoder.decode_event(event_data, event_number, header_dict)

    f_in.close()
    if verbose: update_progress(1)

    if len(unrecognized_data_ids) > 0:
        print("\nGarbage Report!:")
        print("Found the following data IDs, not present in the header:")
        for id in unrecognized_data_ids:
            print("  {}".format(id))
        print("hopefully they weren't important!\n")

    t1_file_name = os.path.join(
        output_dir, output_file_string + '_run{}.h5'.format(runNumber))

    if os.path.isfile(t1_file_name):
        if verbose: print("Over-writing tier1 file {}...".format(t1_file_name))
        os.remove(t1_file_name)

    if verbose:
        print("Writing Tier 1 File:\n    {}".format(t1_file_name))
        for d in decoders:
            print(" -- {}".format(d.decoder_name))
            d.to_file(t1_file_name)
    print("Done.")
