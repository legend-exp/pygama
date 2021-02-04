import numpy as np
import os, re
import pandas as pd
from future.utils import iteritems

from pygama.processing import header_parser
# from header_parser import *
from utils import update_progress
import data_loader as dl

from functools import reduce
import operator


def main():
    print("Beginning test data processing script")

    WF_LEN = 2018
    SEEK_SET = 0
    SEEK_END = 2

    chanList = None  #[600, 601]
    verbose = True

    # filename = "/Users/smeijer/Data/Run827"
    filename = "/Users/smeijer/Downloads/pygama_example/data/2016-3-7-P3KJR_Run11513"
    output_dir = None
    output_file_string = "test"

    directory = os.path.dirname(filename)
    output_dir = os.getcwd() if output_dir is None else output_dir

    #parse the header (in python)
    reclen, reclen2, headerDict = parse_header(filename)

    print("Header parsed.")
    print("   %d longs (in plist header)" % reclen)
    print("   %d bytes in the header" % reclen2)

    f_in = open(filename.encode('utf-8'), "rb")
    if f_in == None:
        print("Couldn't file the file %s" % filename)  # file the file?
        exit(0)

    #figure out the total size
    f_in.seek(0, SEEK_END)
    file_size = float(f_in.tell())
    f_in.seek(0, 0)  # rewind
    file_size_MB = file_size / 1e6
    print("Total file size: %3.3f MB" % file_size_MB)

    # skip the header
    f_in.seek(
        reclen * 4
    )  # reclen is in number of longs, and we want to skip a number of bytes

    # pull out important header info for event processing
    dataIdRun = get_data_id(headerDict, "ORRunModel", "Run")
    dataIdG = get_data_id(headerDict, "ORGretina4M", "Gretina4M")
    runNumber = get_run_number(headerDict)

    id_dict = flip_data_ids(headerDict)

    print("The Data IDs present in this file (header) are: ")
    print("   ", id_dict, "\n")

    # The decoders variable is a list of all the decoders that exist in pygama
    decoders = dl.Data_Loader.get_decoders()
    name_to_id = dict()
    for key in id_dict.keys():
        val = id_dict[key][0]
        name_to_id[val] = key

    # This gives us a list of all the names/ids in our file which we can decode
    decodable_ids = []
    decodable_names = []
    for value in decoders:
        try:
            decodable_ids.append(name_to_id[value])
            decodable_names.append(value)
        except KeyError as e:
            print("There exists a decoder for the", value,
                  ", but no instances of it produced data in this run...")

    print("The available decoders relavent to this file are: ")
    for d in decodable_names:
        print("   ", d, " (ID: ", name_to_id[d], ")")

    print("Additionally, the data ID for the run object is : ", dataIdRun)
    print("Run number: ", runNumber)

    # read all header info into a single, channel-keyed data frame for saving
    headerinfo = get_header_dataframe_info(headerDict)
    df_channels = pd.DataFrame(headerinfo)
    df_channels.set_index("channel", drop=False, inplace=True)
    active_channels = df_channels["channel"].values

    print("Active channels:", active_channels)
    if chanList is not None:
        good_channels = np.ones((len(active_channels)))
        for i, (index, row) in enumerate(df_channels.iterrows()):
            if row.channel not in chanList:
                good_channels[i] = 0
        df_channels = df_channels[good_channels == 1]

    timestamp = 0
    energy = 0
    event_data = np.zeros((20000), dtype='uint32')
    channel = 0
    card = 0
    crate = 0
    board_id = 0

    n = 0  #
    n_max = 100000  # max number of events to process
    res = 0  # result
    board_id_map = {}
    appended_data = []

    wf_data = np.zeros(WF_LEN)

    print("Beginning Tier 0 processing of file {}...".format(filename))

    while (n < n_max and
           f_in.tell() < file_size):  # and f_in.tell() < file_size):
        n = n + 1
        # if verbose and n%1000==0:
        #     update_progress( float(f_in.tell()) / file_size )

        try:
            # print("\nLoading event %d:" % (n))
            event_data, card, crate, data_id = dl.Data_Loader.get_next_event(
                f_in, dataIdRun, dataIdG)
        except EOFError:
            break
        except Exception as e:
            print("Failed to get the next event... (Exception: ", e, ")")
            # print(e)
            break

        if (data_id not in decodable_ids):
            try:
                anId = id_dict[data_id]
                print("No decoder for", anId[0], "(", anId[1],
                      ") device. Skipping...")
            except:
                print(
                    "Data ID of ", data_id,
                    " wasn't in the header dictionary, hopefully it wasn't important"
                )
                pass
            continue

        # print("Decoding event %d:" % (n))

        # Set up my decoders
        g4 = dl.Gretina4m_Decoder()
        mjd = dl.MJDPreamp_Decoder()
        hv = dl.ISegHV_Decoder()

        if (id_dict[data_id][0] == hv.get_name()):
            print("Decoding HV...")
            hv.decode_event(event_data)
            continue

        if (id_dict[data_id][0] == mjd.get_name()):
            print("Decoding MJD Preamp...")
            mjd.decode_event(event_data)
            continue

        if (id_dict[data_id][0] == g4.get_name()):
            # print("Decoding gretina")
            timestamp, energy, channel, wf_data = g4.decode_event(event_data)

        #TODO: this is totally mysterious to me.  why bitshift 9??
        # SJM: it would be 8, but I think for MJD, crate numbers are 1-indexed
        crate_card_chan = (crate << 9) + (card << 4) + (channel)

        if crate_card_chan not in active_channels:
            # print("Data read for channel %d: not an active channel" % crate_card_chan)
            continue
        if chanList is not None and crate_card_chan not in chanList:
            continue

        if crate_card_chan not in board_id_map:
            board_id_map[crate_card_chan] = board_id
        else:
            if not board_id_map[crate_card_chan] == board_id:
                print(
                    "WARNING: previously channel %d had board serial id %d, now it has id %d"
                    % (crate_card_chan, board_id_map[crate_card_chan],
                       board_id))

        #TODO: it feels like the wf can be probabilistically too early or too late in the record?
        #for now, just trim 4 off each side to make length 2010 wfs?
        # SJM: This needs to happen in a raw_to_dsp processing, but not here
        wf_arr = np.array(wf_data, dtype=np.uint16)
        # sig_arr = sig_arr[4:-4]

        data = g4.format_data(energy, timestamp, crate_card_chan, wf_arr)

        # if len(wf_arr) != 966:
        #     print("weird...",len(wf_arr))

        # numpy.isnan(wf_arr).any()

        appended_data.append(data)

    f_in.close()
    if verbose: update_progress(1)
    verbose = True
    if verbose: print("\nCreating dataframe for file {}...".format(filename))
    df_data = pd.DataFrame.from_dict(appended_data)
    t1_file_name = os.path.join(
        output_dir, output_file_string + '_run{}.h5'.format(runNumber))
    if verbose:
        print("Writing {} to raw_to_dsp file {}...".format(filename, t1_file_name))

    board_ids = df_channels['channel'].map(board_id_map)
    df_channels = df_channels.assign(board_id=board_ids)
    # print(df_channels.head())

    df_data.to_hdf(
        t1_file_name,
        key="data",
        mode='w',
        data_columns=['energy', 'channel', 'timestamp'],
        complevel=9)
    df_channels.to_hdf(
        t1_file_name,
        key="channel_info",
        mode='a',
        data_columns=True,
    )

    return df_data


if __name__ == "__main__":
    main()
