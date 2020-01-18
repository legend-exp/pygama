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
from ..io.orca_helper import *
from ..io.llama_3316 import *


def daq_to_raw(t0_file, run, prefix="t1", suffix="", chan_list=None, n_max=np.inf,
               verbose=False, output_dir=None, overwrite=True, decoders=None,
               config={}):
    """
    """
    print("Starting pygama daq_to_raw processing ...")
    print("  Input file:", t0_file)

    output_dir = os.getcwd() if output_dir is None else output_dir

    # ###############################################################
    # # Change for HADES style output
    # # TODO: as vince would say, i need to do something smart here
    #
    # if ftype == "hades_char":
    # # declare Tier 1 output file
    #
    #    file_body = t0_file.split("/")[-1].replace("fcio","h5")
    #    t1_file = "{}/{}_{}".format(output_dir,prefix,file_body)
    #    if os.path.isfile(t1_file):
    #        if overwrite:
    #            print("Overwriting existing file...")
    #            os.remove(t1_file)
    #        else:
    #            print("File already exists, continuing ...")
    #            return
    # ################################################################
    # else:
    t1_file = f"{output_dir}/{prefix}_run{run}.{suffix}"
    if os.path.isfile(t1_file):
       if overwrite:
           print("Overwriting existing file...")
           os.remove(t1_file)
       else:
           print("File already exists, continuing ...")
           return

    t_start = time.time()

    # set max number of events (useful for debugging)
    if n_max is not np.inf and n_max is not None:
        n_max = int(n_max)

    # get the DAQ mode
    if config["daq"] == "ORCA":
        process_orca(t0_file, t1_file, n_max, decoders, config, verbose, run)

    elif config["daq"] == "FlashCam":
        process_flashcam(t0_file, t1_file, run, n_max, decoders, config, verbose)

    elif config["daq"] == "SIS3316":
        process_llama_3316(t0_file, t1_file, run, n_max, config, verbose)

    elif config["daq"] == "CAENDT57XXDecoder":
        process_compass(t0_file, t1_file, decoders, output_dir)

    else:
        print(f"DAQ: {config['daq']} not recognized.  Exiting ...")
        exit()

    # --------- summary ------------

    statinfo = os.stat(t1_file)
    print("File size: {}".format(sizeof_fmt(statinfo.st_size)))
    elapsed = time.time() - t_start
    print("Time elapsed: {:.2f} sec".format(elapsed))
    print("  Output file:", t1_file)
    print("Done.\n")


def process_orca(t0_file, t1_file, n_max, decoders, config, verbose, run=None):
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

    if run is not None:
        run = get_run_number(header_dict)
    print("Run number: {}".format(run))

    # figure out which decoders we can use.  should simplify this block
    decoders = []
    id_dict = get_decoder_for_id(header_dict)
    if verbose:
        print("Data IDs present in this header are:")
        for id in id_dict:
            print(f"    {id}: {id_dict[id]}")
    used_ids = set([id_dict[id] for id in id_dict])
    for sub in DataTaker.__subclasses__():
        tmp = sub() # instantiate the class
        if tmp.decoder_name in used_ids:
            # tmp.apply_config(config) # broken rn
            decoders.append(tmp)
    decoder_to_id = {d.decoder_name: d for d in decoders}
    if verbose:
        print("pygama will run these decoders:")
        for d in decoders:
            print("   ", d.decoder_name)

    # -- scan over raw data --
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

    print("Wrote Tier 1 File:\n    {}\nFILE INFO:".format(t1_file))
    with pd.HDFStore(t1_file,'r') as store:
        print(store.keys())
        # print(store.info())


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

    pprint(header_dict)

    #see pygama/pygama/io/decoders/io_base.py
    decoders = []
    #decoders.append(LLAMAStruck3316(metadata=pd.DataFrame.from_dict(header_dict)))   #we just have that one
    decoders.append(LLAMAStruck3316(metadata=header_dict))  #we just have that one
                    # fix: saving metadata using io_bases ctor
                    # have to convert to dataframe here in order to avoid
                    # passing to xml_header.get_object_info in io_base.load_metadata
    channelOne = list(list(header_dict.values())[0].values())[0]
    decoders[0].initialize(1000./channelOne["SampleFreq"], channelOne["Gain"])
        # FIXME: gain set according to first found channel, but gain can change!

    print("pygama will run this fancy decoder: SIS3316Decoder")

    # pass in specific decoder options (windowing, multisampling, etc.)
    #for d in decoders:
    #    d.apply_config(config) #no longer used (why?)

    # ------------ scan over raw data starts here -----------------
    # more code duplication

    print("Beginning Tier 0 processing ...")

    packet_id = 0  # number of events decoded
    row_id = 0      #index of written rows, FIXME maybe gets unused
    unrecognized_data_ids = []

    # header is already skipped by llama_3316,

    def toFile(digitizer, filename_raw, rowID, verbose):
        numb = str(rowID).zfill(4)
        filename_mod = filename_raw + "." + numb
        print("redirecting output file to packetfile "+filename_mod)
        digitizer.save_to_pytables(filename_mod, verbose)
    

    # start scanning
    while (packet_id < n_max and f_in.tell() < file_size):
        packet_id += 1

        if verbose and packet_id % 1000 == 0:
            update_progress(float(f_in.tell()) / file_size)

        # write periodically to the output file instead of writing all at once
        if packet_id % ROW_LIMIT == 0:
            for d in decoders:
                d.save_to_lh5(t1_file)
            row_id += 1

        try:
            fadcID, channelID, event_data = sisfile.read_next_event(header_dict)
        except Exception as e:
            print("Failed to get the next event ... Exception:",e)
            break
        if event_data is None:
            break

        decoder = decoders[0]       #well, ...
        # sends data to the pandas dataframe
        decoder.decode_event(event_data, packet_id, header_dict, fadcID, channelID)

    print("done.  last packet ID:", packet_id)
    f_in.close()

    # final write to file
    for d in decoders:
        d.save_to_lh5(t1_file)

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
    #    # print(store.info())
    


def process_compass(t0_file, t1_file, digitizer, output_dir=None):
    """
    Takes an input .bin file name as t0_file from CAEN CoMPASS and outputs t1_file
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
    i_debug = 0
    packet_id = 0
    while fcio.next_event() and packet_id < n_max:
      packet_id += 1
      if verbose and packet_id % 1000 == 0:
          update_progress(float(fcio.telid) / file_size)

      # write periodically to the output file
      if packet_id % ROW_LIMIT == 0:

          # decoder.save_to_pytables(t1_file, verbose=True)

          decoder.save_to_lh5(t1_file)

          i_debug += 1
          if i_debug == 1:
              print("breaking early")
              break # debug, deleteme

      decoder.decode_event(fcio, packet_id)

    # end of loop, write to file once more
    # decoder.save_to_pytables(t1_file, verbose=True)
    # decoder.save_to_lh5(t1_file, verbose=True)

    if verbose:
      update_progress(1)
