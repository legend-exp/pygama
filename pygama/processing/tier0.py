""" ========= PYGAMA =========
TIER 0 MAIN PROCESSING ROUTINE
"""
# cimport numpy as np

import numpy as np
import os, re, sys, glob, time
import pandas as pd
import h5py
from future.utils import iteritems
from functools import partial

from ..utils import *
from ..decoders.digitizers import *
from ..decoders.data_loading import *
from ..decoders.xml_parser import *
from .base import *


def ProcessTier0(filename,
                 output_file_string="t1",
                 chan_list=None,
                 n_max=np.inf,
                 verbose=False,
                 output_dir=None,
                 decoders=None,
                 flatten=False):
    """
    Reads in "raw / tier 0" ORCA data and saves to an hdf5 format using pandas
    filename: path to an orca data file
    output_file_string: output file name will be <output_file_string>_run<runNumber>.h5
    n_max: maximum number of events to process (useful for debugging)
    verbose: spits out a progressbar to let you know how the processing is going
    output_dir: where to stash the t1 file
    """
    print("Starting pygama Tier 0 processing ...")
    print("  Input file: "+filename)
    start = time.clock()

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

    # figure out the total size
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
            d.to_file(t1_file_name, flatten)

        statinfo = os.stat(t1_file_name)
        print("File size: {}".format(sizeof_fmt(statinfo.st_size)))
        elapsed = time.clock() - start
        print("Time elapsed: {:.2f} sec".format(elapsed))
        print("Done.\n")


def get_next_event(f_in):
    """
    Gets the next event, and some basic information about it
    Takes the file pointer as input
    Outputs:
    -event_data: a byte array of the data produced by the card (could be header + data)
    -slot:
    -crate:
    -data_id: This is the identifier for the type of data-taker (i.e. Gretina4M, etc)
    """
    # number of bytes to read in = 8 (2x 32-bit words, 4 bytes each)

    # The read is set up to do two 32-bit integers, rather than bytes or shorts
    # This matches the bitwise arithmetic used elsewhere best, and is easy to implement
    # Using a

    # NCRATES = 10

    try:
        head = np.fromstring(
            f_in.read(4), dtype=np.uint32)  # event header is 8 bytes (2 longs)
    except Exception as e:
        print(e)
        raise Exception("Failed to read in the event orca header.")

    # Assuming we're getting an array of bytes:
    # record_length   = (head[0] + (head[1]<<8) + ((head[2]&0x3)<<16))
    # data_id         = (head[2] >> 2) + (head[3]<<8)
    # slot            = (head[6] & 0x1f)
    # crate           = (head[6]>>5) + head[7]&0x1
    # reserved        = (head[4] + (head[5]<<8))

    # Using an array of uint32
    record_length = int((head[0] & 0x3FFFF))
    data_id = int((head[0] >> 18))
    # slot            =int( (head[1] >> 16) & 0x1f)
    # crate           =int( (head[1] >> 21) & 0xf)
    # reserved        =int( (head[1] &0xFFFF))

    # /* ========== read in the rest of the event data ========== */
    try:
        event_data = f_in.read(record_length * 4 -
                               4)  # record_length is in longs, read gives bytes
    except Exception as e:
        print("  No more data...\n")
        print(e)
        raise EOFError

    # if (crate < 0 or crate > NCRATES or slot  < 0 or slot > 20):
    #     print("ERROR: Illegal VME crate or slot number {} {} (data ID {})".format(crate, slot,data_id))
    #     raise ValueError("Encountered an invalid value of the crate or slot number...")

    # return event_data, slot, crate, data_id
    return event_data, data_id


def get_decoders(object_info):
    """
    Find all the active pygama data takers that inherit from the DataLoader class.
    This only works if the subclasses have been imported.  Is that what we want?
    Also relies on 2-level abstraction, which is dicey
    """
    decoders = []
    for sub in DataLoader.__subclasses__():  # either digitizers or pollers
        for subsub in sub.__subclasses__():
            try:
                decoder = subsub(object_info) # initialize the decoder
                # print("dataloading - name: ",decoder.decoder_name)
                decoders.append(decoder)
            except Exception as e:
                print(e)
                pass
    return decoders
