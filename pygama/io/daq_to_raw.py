"""
pygama tier 0 processing
raw daq data --> pandas dfs saved to hdf5 file (tier 1)
"""
import os, re, sys, glob, time
import numpy as np
import h5py
import pandas as pd
from pprint import pprint

from ..utils import *
from ..io.digitizers import *
from ..io.pollers import *
from ..io.io_base import *
from ..io.orca_header import *
from ..io.llama_3316 import *             


def daq_to_raw(t0_file, run, output_prefix="t1", chan_list=None, n_max=np.inf,
               verbose=False, output_dir=None, overwrite=True, decoders=None,
               config={}):
    """
    """
    print("Starting pygama daq_to_raw processing ...")
    print("  Input file:", t0_file)

    # declare Tier 1 output file
    output_dir = os.getcwd() if output_dir is None else output_dir
    t1_file = "{}/{}_run{}.h5".format(output_dir, output_prefix, run)
    if os.path.isfile(t1_file):
        if overwrite:
            print("Overwriting existing file...")
            os.remove(t1_file)
        else:
            print("File already exists, continuing ...")
            return
        
    # set max number of events (useful for debugging)
    if n_max is not np.inf and n_max is not None:
        n_max = int(n_max)

    # get the DAQ mode
    if config["daq"] == "ORCA":
        process_orca(t0_file, t1_file, run, n_max, decoders, config, verbose)
    
    elif config["daq"] == "FlashCam":
        process_flashcam(t0_file, t1_file, run, n_max, decoders, config, verbose)
    
    elif config["daq"] == "SIS3316":
        process_llama_3316(t0_file, t1_file, run, n_max, config, verbose)
    
    elif config["daq"] == "CAENDT57XXDecoder":
        process_compass(t0_file, t1_file, decoders, output_dir)
    
    else:
        print(f"DAQ: {config['daq']} not recognized.  Exiting ...")
        exit()
    
    
def process_orca(t0_file, t1_file, run, n_max, decoders, config, verbose):
    """
    convert ORCA DAQ data to pygama "raw" lh5
    """
    
    
    ROW_LIMIT = 5e4
    
    start = time.time()
    f_in = open(t0_file.encode('utf-8'), "rb")
    if f_in == None:
        print("Couldn't find the file %s" % t0_file)
        sys.exit(0)

    # parse the header
    reclen, reclen2, header_dict = parse_header(t0_file)

    # figure out the total size
    SEEK_END = 2
    f_in.seek(0, SEEK_END)
    file_size = float(f_in.tell())
    f_in.seek(0, 0)  # rewind
    file_size_MB = file_size / 1e6
    print("Total file size: {:.3f} MB".format(file_size_MB))

    # run = get_run_number(header_dict)
    print("Run number: {}".format(run))

    id_dict = get_decoder_for_id(header_dict)
    if verbose:
        print("Data IDs present in this header are:")
        for id in id_dict:
            print("    {}: {}".format(id, id_dict[id]))
    used_decoder_names = set([id_dict[id] for id in id_dict])

    # get all available pygama decoders, then remove unused ones
    if decoders is None:
        decoders = get_decoders(header_dict)
        decoder_names = [d.decoder_name for d in decoders]

    final_decoder_list = list(set(decoder_names).intersection(used_decoder_names))
    decoders = [d for d in decoders if d.decoder_name in final_decoder_list]
    decoder_to_id = {d.decoder_name: d for d in decoders}

    print("pygama will run these decoders:")
    for name in final_decoder_list:
        for id in id_dict:
            if id_dict[id] == name:
                this_data_id = id
        print("    {}: {}".format(this_data_id, name))

    # pass in specific decoder options (windowing, multisampling, etc.)
    for d in decoders:
        d.apply_config(config)

    
    # ------------ scan over raw data starts here -----------------

    print("Beginning Tier 0 processing ...")

    packet_id = 0  # number of events decoded
    unrecognized_data_ids = []

    # skip the header.
    # reclen is in number of longs, and we want to skip a number of bytes
    f_in.seek(reclen * 4)

    # start scanning
    while (packet_id < n_max and f_in.tell() < file_size):
        packet_id += 1

        if verbose and packet_id % 1000 == 0:
            update_progress(float(f_in.tell()) / file_size)

        # write periodically to the output file instead of writing all at once
        if packet_id % ROW_LIMIT == 0:
            for d in decoders:
                d.save_to_pytables(t1_file, verbose=True)

        try:
            event_data, data_id = get_next_event(f_in)
        except EOFError:
            break
        except Exception as e:
            print("Failed to get the next event ... Exception:",e)
            break
        try:
            decoder = decoder_to_id[id_dict[data_id]]
        except KeyError:
            if data_id not in id_dict and data_id not in unrecognized_data_ids:
                unrecognized_data_ids.append(data_id)
            continue

        # sends data to the pandas dataframe
        decoder.decode_event(event_data, packet_id, header_dict)

    print("done.  last packet ID:", packet_id)
    f_in.close()

    # final write to file
    for d in decoders:
        d.save_to_pytables(t1_file, verbose=True)

    if verbose:
        update_progress(1)

    if len(unrecognized_data_ids) > 0:
        print("WARNING, Found the following unknown data IDs:")
        for id in unrecognized_data_ids:
            print("  {}".format(id))
        print("hopefully they weren't important!\n")

    # ---------  summary ------------

    print("Wrote: Tier 1 File:\n    {}\nFILE INFO:".format(t1_file))
    with pd.HDFStore(t1_file,'r') as store:
        print(store.keys())
        # print(store.info())

    statinfo = os.stat(t1_file)
    print("File size: {}".format(sizeof_fmt(statinfo.st_size)))
    elapsed = time.time() - start
    print("Time elapsed: {:.2f} sec".format(elapsed))
    print("Done.\n")


def get_next_event(f_in):
    """
    Gets the next event, and some basic information about it
    Takes the file pointer as input
    Outputs:
    -event_data: a byte array of the data produced by the card (could be header + data)
    -data_id: This is the identifier for the type of data-taker (i.e. Gretina4M, etc)
    # number of bytes to read in = 8 (2x 32-bit words, 4 bytes each)
    # The read is set up to do two 32-bit integers, rather than bytes or shorts
    # This matches the bitwise arithmetic used elsewhere best, and is easy to implement
    """
    try:
        # event header is 8 bytes (2 longs)
        head = np.fromstring(f_in.read(4), dtype=np.uint32)  
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
        # record_length is in longs, read gives bytes
        event_data = f_in.read(record_length * 4 - 4)
    except Exception as e:
        print("  No more data...\n")
        print(e)
        raise EOFError

    return event_data, data_id
    

def process_llama_3316(t0_file, t1_file, run, n_max, config, verbose):
    """
    convert llama DAQ data to pygama "raw" lh5
    
    Mario's implementation for the Struck SIS3316 digitizer.
    Requires the llamaDAQ program for producing compatible input files.
    """       
    ROW_LIMIT = 5e4
    
    start = time.time()
    f_in = open(t0_file.encode('utf-8'), "rb")
    if f_in == None:
        print("Couldn't find the file %s" % t0_file)
        sys.exit(0)
        
    #file = llama_3316(f_in,2) #test

    verbosity = 1 if verbose else 0     # 2 is for debug
    sisfile = llama_3316(f_in, verbosity)

    # figure out the total size
    SEEK_END = 2
    f_in.seek(0, SEEK_END)
    file_size = float(f_in.tell())
    f_in.seek(0, 0)  # rewind
    file_size_MB = file_size / 1e6
    print("Total file size: {:.3f} MB".format(file_size_MB))
    
    header_dict = sisfile.parse_channelConfigs()    # parse the header dict after manipulating position in file

    # run = get_run_number(header_dict)
    print("Run number: {}".format(run))

    #see pygama/pygama/io/decoders/io_base.py
    decoders = []
    decoders.append(SIS3316Decoder(pd.DataFrame.from_dict(header_dict)))   #we just have that one
                    # fix: saving metadata using io_bases ctor
                    # have to convert to dataframe here in order to avoid 
                    # passing to xml_header.get_object_info in io_base.load_metadata
    channelOne = list(list(header_dict.values())[0].values())[0]
    decoders[0].initialize(1000./channelOne["SampleFreq"], channelOne["Gain"])
        # FIXME: gain set according to first found channel, but gain can change!

    print("pygama will run this fancy decoder: SIS3316Decoder")

    # pass in specific decoder options (windowing, multisampling, etc.)
    for d in decoders:
        d.apply_config(config)

    # ------------ scan over raw data starts here -----------------
    # more code duplication

    print("Beginning Tier 0 processing ...")

    packet_id = 0  # number of events decoded
    unrecognized_data_ids = []

    # header is already skipped by llama_3316

    # start scanning
    while (packet_id < n_max and f_in.tell() < file_size):
        packet_id += 1

        if verbose and packet_id % 1000 == 0:
            update_progress(float(f_in.tell()) / file_size)

        # write periodically to the output file instead of writing all at once
        if packet_id % ROW_LIMIT == 0:
            for d in decoders:
                d.save_to_pytables(t1_file, verbose=True)

        try:
            fadcID, channelID, event_data = sisfile.read_next_event(header_dict)
        except Exception as e:
            print("Failed to get the next event ... Exception:",e)
            break
        if event_data is None:      #EOF
            break
            
        decoder = decoders[0]       #well, ...
        # sends data to the pandas dataframe
        decoder.decode_event(event_data, packet_id, header_dict, fadcID, channelID)

    print("done.  last packet ID:", packet_id)
    f_in.close()

    # final write to file
    for d in decoders:
        d.save_to_pytables(t1_file, verbose=True)

    if verbose:
        update_progress(1)

    if len(unrecognized_data_ids) > 0:
        print("WARNING, Found the following unknown data IDs:")
        for id in unrecognized_data_ids:
            print("  {}".format(id))
        print("hopefully they weren't important!\n")

    # ---------  summary ------------

    print("Wrote: Tier 1 File:\n    {}\nFILE INFO:".format(t1_file))
    with pd.HDFStore(t1_file,'r') as store:
        print(store.keys())
        # print(store.info())

    statinfo = os.stat(t1_file)
    print("File size: {}".format(sizeof_fmt(statinfo.st_size)))
    elapsed = time.time() - start
    print("Time elapsed: {:.2f} sec".format(elapsed))
    print("Done.\n")


def process_compass(t0_file, t1_file, digitizer, output_dir=None):
    """
    Takes an input .bin file name as t0_file from CAEN CoMPASS and outputs a .h5 file named t1_file

    t0_file: input file name, string type
    t1_file: output file name, string type
    digitizer: CAEN digitizer, Digitizer type
    options are uncalibrated or calibrated.  Select the one that was outputted by CoMPASS, string type
    output_dir: path to output directory string type

    """
    start = time.time()
    f_in = open(t0_file.encode("utf-8"), "rb")
    if f_in is None:
        raise LookupError("Couldn't find the file %s" % t0_file)
    SEEK_END = 2
    f_in.seek(0, SEEK_END)
    file_size = float(f_in.tell())
    f_in.seek(0, 0)
    file_size_MB = file_size / 1e6
    print("Total file size: {:.3f} MB".format(file_size_MB))

    # ------------- scan over raw data starts here ----------------

    print("Beginning Tier 0 processing ...")

    event_rows = []
    waveform_rows = []
    event_size = digitizer.get_event_size(t0_file)
    with open(t0_file, "rb") as metadata_file:
        event_data_bytes = metadata_file.read(event_size)
        while event_data_bytes != b"":
            event, waveform = digitizer.get_event(event_data_bytes)
            event_rows.append(event)
            waveform_rows.append(waveform)
            event_data_bytes = metadata_file.read(event_size)
    all_data = np.concatenate((event_rows, waveform_rows), axis=1)
    output_dataframe = digitizer.create_dataframe(all_data)
    f_in.close()

    output_dataframe.to_hdf(path_or_buf=output_dir+"/"+t1_file, key="dataset", mode="w", table=True)
    print("Wrote Tier 1 File:\n  {}\nFILE INFO:".format(t1_file))

    # --------- summary -------------

    with pd.HDFStore(t1_file, "r") as store:
        print(store.keys())

    statinfo = os.stat(t1_file)
    print("File size: {}".format(sizeof_fmt(statinfo.st_size)))
    elapsed = time.time() - start
    print("Time elapsed: {:.2f} sec".format(elapsed))
    print("Done.\n")


def process_flashcam(t0_file, t1_file, run, n_max, decoders, config, verbose):
    """
    decode FlashCam data, using the fcutils package to handle file access,
    and the FlashCam DataTaker to save the results and write to output.
    """
    import fcutils

    fcio = fcutils.fcio(t0_file)
    decoder = FlashCam()
    decoder.get_file_config(fcio)
    
    # ROW_LIMIT = 5e4
    ROW_LIMIT = 1000

    # loop over raw data packets
    packet_id = 0
    while fcio.next_event() and packet_id < n_max:
      packet_id += 1
      if verbose and packet_id % 1000 == 0:
          update_progress(float(fcio.telid) / file_size)
      
      # write periodically to the output file
      if packet_id % ROW_LIMIT == 0:
          # decoder.save_to_pytables(t1_file, verbose=True)
          decoder.save_to_lh5(t1_file, verbose=True)
          break # debug, deleteme

      decoder.decode_event(fcio, packet_id)

    # end of loop, write to file once more
    decoder.save_to_pytables(t1_file, verbose=True)

    if verbose:
      update_progress(1)

    # --------- summary ------------

    statinfo = os.stat(t1_file)
    print("File size: {}".format(sizeof_fmt(statinfo.st_size)))
    elapsed = time.time() - start
    print("Time elapsed: {:.2f} sec".format(elapsed))
    print("Done.\n")
