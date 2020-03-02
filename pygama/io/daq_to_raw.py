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


def daq_to_raw(daq_filename, raw_filename=None, run=None, prefix="t1", suffix="lh5", chan_list=None, n_max=np.inf,
               verbose=False, output_dir=None, overwrite=True, decoders=None,
               config={}):
    """
    """
    print("Starting pygama daq_to_raw processing ...")
    print("  Input file:", daq_filename)

    if raw_filename is None:
        if run is None:
            print('daq_to_raw error: must supply either raw_filename or run number')
            sys.exit()
        output_dir = os.getcwd() if output_dir is None else output_dir
        raw_filename = f"{output_dir}/{prefix}_run{run}.{suffix}"

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

    t_start = time.time()

    # set max number of events (useful for debugging)
    if n_max is not np.inf and n_max is not None:
        n_max = int(n_max)

    # get the DAQ mode
    if config["daq"] == "ORCA":
        process_orca(daq_filename, raw_filename, n_max, decoders, config, verbose, run)

    elif config["daq"] == "FlashCam":
        process_flashcam(daq_filename, raw_filename, n_max, config, verbose)

    elif config["daq"] == "SIS3316":
        process_llama_3316(daq_filename, raw_filename, run, n_max, config, verbose)

    elif config["daq"] == "CAENDT57XXDecoder":
        process_compass(daq_filename, raw_filename, decoders, output_dir)

    else:
        print(f"DAQ: {config['daq']} not recognized.  Exiting ...")
        exit()

    # --------- summary ------------

    statinfo = os.stat(raw_filename)
    print("File size: {}".format(sizeof_fmt(statinfo.st_size)))
    elapsed = time.time() - t_start
    print("Time elapsed: {:.2f} sec".format(elapsed))
    print("  Output file:", raw_filename)
    print("Done.\n")


def process_orca(daq_filename, raw_filename, n_max, decoders, config, verbose, run=None):
    """
    convert ORCA DAQ data to "raw" lh5
    """
    buffer_size = 1024
    #buffer_size = 50000

    lh5_store = LH5Store()

    f_in = open(daq_filename.encode('utf-8'), "rb")
    if f_in == None:
        print("Couldn't find the file %s" % daq_filename)
        sys.exit(0)

    # parse the header. save the length so we can jump past it later
    reclen, reclen2, header_dict = parse_header(daq_filename)

    # figure out the total size
    SEEK_END = 2
    f_in.seek(0, SEEK_END)
    file_size = float(f_in.tell())
    f_in.seek(0, 0)  # rewind
    file_size_MB = file_size / 1e6
    print("Total file size: {:.3f} MB".format(file_size_MB))

    if run is not None:
        run = get_run_number(header_dict)
    print("Run number:", run)

    # figure out which decoders we can use and build a dictionary from 
    # dataID to decoder
    decoders = {}
    id2dn_dict = get_id_to_decoder_name_dict(header_dict)
    if verbose:
        print("Data IDs present in ORCA file header are:")
        for ID in id2dn_dict:
            print(f"    {ID}: {id2dn_dict[ID]}")
    dn2id_dict = {}
    for ID, name in id2dn_dict.items(): dn2id_dict[name] = ID
    for sub in OrcaDecoder.__subclasses__():
        decoder = sub() # instantiate the class
        if decoder.decoder_name in dn2id_dict:
            # Later: allow to turn on / off decoders in exp.json
            if decoder.decoder_name != 'ORSIS3302DecoderForEnergy': continue
            decoder.dataID = dn2id_dict[decoder.decoder_name]
            decoder.set_object_info(get_object_info(header_dict, decoder.orca_class_name))
            decoders[decoder.dataID] = decoder
    if verbose:
        print("pygama will run these decoders:")
        for ID, dec in decoders.items():
            print("   ", dec.decoder_name+ ", id =", ID)

    # Set up dataframe buffers -- for now, one for each decoder
    # Later: control in intercom
    tbs = {}
    for data_id, dec in decoders.items():
        tbs[data_id] = LH5Table(buffer_size)
        dec.initialize_lh5_table(tbs[data_id])

    # -- scan over raw data --
    print("Beginning Tier 0 processing ...")

    packet_id = 0  # number of events decoded
    unrecognized_data_ids = []

    # skip the header using reclen from before
    # reclen is in number of longs, and we want to skip a number of bytes
    f_in.seek(reclen * 4)

    # start scanning
    while (packet_id < n_max and f_in.tell() < file_size):
        packet_id += 1

        if verbose and packet_id % 1000 == 0:
            update_progress(float(f_in.tell()) / file_size)

        try:
            packet, data_id = get_next_packet(f_in)
        except EOFError:
            break
        except Exception as e:
            print("Failed to get the next event ... Exception:", e)
            break

        if data_id not in decoders:
            if data_id not in unrecognized_data_ids: 
                unrecognized_data_ids.append(data_id)
            continue

        if data_id in tbs.keys():
            tb = tbs[data_id]
            decoder.decode_packet(packet, tb, packet_id, header_dict)
            # write to the output file when a buffer gets full
            if tb.is_full():
                lh5_store.write_object(tb, id2dn_dict[data_id], raw_filename, n_rows=tb.size)
                tb.clear()


    print("done.  last packet ID:", packet_id)
    f_in.close()

    # final write to file
    for data_id in tbs.keys():
        tb = tbs[data_id]
        if tb.loc == 0: continue
        lh5_store.write_object(tb, id2dn_dict[data_id], raw_filename, n_rows=tb.loc)
        tb.clear()

    if verbose:
        update_progress(1)

    if len(unrecognized_data_ids) > 0:
        print("WARNING, Found the following unknown data IDs:")
        for id in unrecognized_data_ids:
            print("  {}".format(id))
        print("hopefully they weren't important!\n")

    print("Wrote Tier 1 File:\n    {}\nFILE INFO:".format(raw_filename))



def process_llama_3316(daq_filename, raw_filename, run, n_max, config, verbose):
    """
    convert llama DAQ data to pygama "raw" lh5

    Mario's implementation for the Struck SIS3316 digitizer.
    Requires the llamaDAQ program for producing compatible input files.
    """
    ROW_LIMIT = 5e4

    start = time.time()
    f_in = open(daq_filename.encode('utf-8'), "rb")
    if f_in == None:
        print("Couldn't find the file %s" % daq_filename)
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
                d.save_to_lh5(raw_filename)
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
        d.save_to_lh5(raw_filename)

    if verbose:
        update_progress(1)

    if len(unrecognized_data_ids) > 0:
        print("WARNING, Found the following unknown data IDs:")
        for id in unrecognized_data_ids:
            print("  {}".format(id))
        print("hopefully they weren't important!\n")

    # ---------  summary ------------

    print("Wrote: Tier 1 File:\n    {}\nFILE INFO:".format(raw_filename))
    with pd.HDFStore(raw_filename,'r') as store:
        print(store.keys())
    #    # print(store.info())
    


def process_compass(daq_filename, raw_filename, digitizer, output_dir=None):
    """
    Takes an input .bin file name as daq_filename from CAEN CoMPASS and outputs raw_filename
    daq_filename: input file name, string type
    raw_filename: output file name, string type
    digitizer: CAEN digitizer, Digitizer type
    options are uncalibrated or calibrated.  Select the one that was outputted by CoMPASS, string type
    output_dir: path to output directory string type
    """
    start = time.time()
    f_in = open(daq_filename.encode("utf-8"), "rb")
    if f_in is None:
        raise LookupError("Couldn't find the file %s" % daq_filename)
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
    event_size = digitizer.get_event_size(daq_filename)
    with open(daq_filename, "rb") as metadata_file:
        event_data_bytes = metadata_file.read(event_size)
        while event_data_bytes != b"":
            event, waveform = digitizer.get_event(event_data_bytes)
            event_rows.append(event)
            waveform_rows.append(waveform)
            event_data_bytes = metadata_file.read(event_size)
    all_data = np.concatenate((event_rows, waveform_rows), axis=1)
    output_dataframe = digitizer.create_dataframe(all_data)
    f_in.close()

    output_dataframe.to_hdf(path_or_buf=output_dir+"/"+raw_filename, key="dataset", mode="w", table=True)
    print("Wrote Tier 1 File:\n  {}\nFILE INFO:".format(raw_filename))

    # --------- summary -------------

    with pd.HDFStore(raw_filename, "r") as store:
        print(store.keys())


def process_flashcam(daq_filename, raw_filename, n_max, config, verbose):
    """
    decode FlashCam data, using the fcutils package to handle file access,
    and the FlashCam DataTaker to save the results and write to output.
    """
    import fcutils

    buffer_size = 1024
    #buffer_size = 5e4

    file_size = os.path.getsize(daq_filename)
    fcio = fcutils.fcio(daq_filename)

    event_decoder = FlashCamEventDecoder()
    event_decoder.get_file_config(fcio)
    event_tb = LH5Table(buffer_size)
    event_decoder.initialize_lh5_table(event_tb)

    status_decoder = FlashCamStatusDecoder()
    status_decoder.get_file_config(fcio)
    status_tb = LH5Table(buffer_size)
    status_decoder.initialize_lh5_table(status_tb)

    # TODO: add overwrite capability
    lh5_store = LH5Store()

    # loop over raw data packets
    i_debug = 0
    packet_id = 0
    rc=1
    while rc and packet_id < n_max:
        rc = fcio.get_record()

        # Skip non-interesting records
        # FIXME: push to a buffer of skipped packets?
        if rc == 0 or rc == 1 or rc == 2 or rc == 5: continue 

        packet_id += 1
        if verbose and packet_id % 1000 == 0:
            # FIXME: is cast to float necessary?
            update_progress(float(fcio.telid) / file_size)

        if rc == 4: # Status record
            status_decoder.decode_packet(fcio, status_tb, packet_id)
            if status_tb.is_full():
                lh5_store.write_object(status_tb, 'fcstatus', raw_filename, n_rows=stats_tb.size)
                status_tb.clear()

        if rc == 3 or rc == 6: # Event or SparseEvent record
            event_decoder.decode_packet(fcio, event_tb, packet_id)
            if event_tb.is_full():
                lh5_store.write_object(event_tb, 'daqdata', raw_filename, n_rows=event_tb.size)
                event_tb.clear()

                #i_debug += 1
                #if i_debug == 1:
                #    print("breaking early")
                #    break # debug, deleteme


    # end of loop, write to file once more
    # decoder.save_to_pytables(raw_filename, verbose=True)
    if event_tb.loc != 0:
        lh5_store.write_object(event_tb, 'daqdata', raw_filename, n_rows=event_tb.loc)
        event_tb.clear()
    if status_tb.loc != 0:
        lh5_store.write_object(status_tb, 'daqdata', raw_filename, n_rows=status_tb.loc)
        status_tb.clear()

    if verbose: update_progress(1)

