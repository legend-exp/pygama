"""
pygama tier 1 processing:
    tier 1 data
    --> DSP -->
    tier 2 (i.e. gatified)
"""
import os, re, sys, time
import numpy as np
import pandas as pd
import datetime
import multiprocessing as mp
from functools import partial

from ..io.decoders.data_loading import *
from ..io.decoders.digitizers import *
from ..utils import *


def ProcessTier1(t1_file,
                 intercom,
                 digitizers=None,
                 ftype="default",
                 output_dir=None,
                 output_prefix="t2",
                 overwrite=True,
                 verbose=False,
                 nevt=None,
                 ioff=None,
                 multiprocess=True,
                 chunk=3000):

    print("Starting pygama Tier 1 processing ...")
    print("  Date:", datetime.datetime.now())
    print("  Input file:", t1_file)
    print("  Size: ", sizeof_fmt(os.path.getsize(t1_file)))
    t_start = time.time()

    # multiprocessing parameters
    CHUNKSIZE = chunk
    ncpu = mp.cpu_count()

    start = time.time()
    in_dir = os.path.dirname(t1_file)
    output_dir = os.getcwd() if output_dir is None else output_dir

    # snag the run number (assuming t1_file ends in _run<number>.<filetype>)
    run_str = re.findall('run\d+', t1_file)[-1]
    run = int(''.join(filter(str.isdigit, run_str)))

    # declare output file
    output_dir = os.getcwd() if output_dir is None else output_dir


    #################################################################
    """
    Change for HADES style data structre
    """ 
    if ftype == "hades_char":
       file_body = t1_file.split("/")[-1].replace("t1","t2")
       t2_file = "{}/{}".format(output_dir, file_body)
    #################################################################
    
    else:
       t2_file = os.path.join(output_dir, "t2_run{}.h5".format(run))

    if os.path.isfile(t2_file):
        if overwrite:
            print("Overwriting existing file...")
            os.remove(t2_file)
        else:
            print("File already exists, continuing ...")
            return

    # get digitizers
    if digitizers is None:
        decoders = get_decoders()
        digitizers = [d for d in decoders if isinstance(d, Digitizer)]
        with pd.HDFStore(t1_file, 'r') as store:
            keys = [key[1:] for key in store.keys()]  # remove leading '/'
            digitizers = [d for d in digitizers if d.decoder_name in keys]

    # go running
    for d in digitizers:
        print("Processing data from digitizer: {}".format(d.decoder_name))

        # get some info about the objects in the file
        with pd.HDFStore(t1_file, 'r') as store:

            s = store.get_storer(d.decoder_name)
            object_info = store.get(d.class_name)
            d.load_metadata(object_info)

            if isinstance(s, pd.io.pytables.AppendableFrameTable):
                use_pytables = True
                nrows = s.nrows
                chunk_idxs = list(range(nrows // CHUNKSIZE + 1))
                nchunks = len(chunk_idxs)
            elif isinstance(s, pd.io.pytables.FrameFixed):
                use_pytables = False
            else:
                print("Unknown type!", type(s))
                exit()

        # --------------- run multiprocessing ----------------
        if use_pytables and multiprocess:

            print("Found {} rows, splitting into {} chunks".format(
                nrows, nchunks))

            keywords = {
                "t1_file": t1_file,
                "chunksize": CHUNKSIZE,
                "nchunks": nchunks,
                "ncpu": ncpu,
                "key": d.decoder_name,
                "intercom": intercom,
                "verbose": verbose
            }

            global ichunk, pstart
            ichunk, pstart = 0, time.time()
            with mp.Pool(ncpu) as p:
                result_list = p.map(
                    partial(process_chunk, **keywords), chunk_idxs)

            # debug: process chunks in series
            # for idx in chunk_idxs:
            # process_chunk(idx, **keywords)
            # exit()

            t2_df = pd.concat(result_list)

        # ---------------- single process data ----------------
        # if df is fixed, we have to read the whole thing in
        else:
            print("WARNING: no multiprocessing")

            if nevt is not np.inf:
                print("limiting to {} events".format(nevt))

                # print("nevt ioff", nevt, ioff)
                nevt = int(nevt)
                if ioff != 0:
                    ioff = int(ioff)

                t1_df = pd.read_hdf(
                    t1_file,
                    key=d.decoder_name,
                    where = "ievt > {} & ievt < {}".format(ioff, ioff+nevt))
            else:
                print("WARNING: no event limit set (-n option)")
                print("read the whole df into memory?  are you sure? (y/n)")
                if input() == "y":
                    t1_df = pd.read_hdf(t1_file, key=d.decoder_name)
                else:
                    exit()

            t2_df = intercom.process(t1_df, verbose)

    update_progress(1)

    # ---------------- write Tier 2 output ----------------

    print("Writing Tier 2 File:\n   {}".format(t2_file))
    print("  Entries: {}".format(len(t2_df)))

    if verbose:
        print("  Data columns:\n", t2_df.columns.values)

    t2_df.to_hdf(
        t2_file,
        key="data",
        format='table',
        mode='w',
        data_columns=t2_df.columns.tolist(),
        complib="blosc:snappy",
        complevel=2
        )

    statinfo = os.stat(t2_file)
    print("  File size: {}".format(sizeof_fmt(statinfo.st_size)))
    elapsed = time.time() - start
    proc_rate = elapsed / len(t2_df)
    print("  Date:", datetime.datetime.now())
    print("  Time elapsed: {:.2f} min  ({:.5f} sec/wf)".format(
        elapsed / 60, proc_rate))
    print("Done.\n")


def process_chunk(chunk_idx,
                  t1_file,
                  chunksize,
                  nchunks,
                  ncpu,
                  key,
                  intercom,
                  verbose=False):
    """
    use hdf5 indexing, which is way faster than reading in the df first.
    this is a really good reason to use the 'tables' format
    """
    # check progress
    # check: i estimated 6 mins from 11:19
    # if it's x4, then that's 6/4 = 1.5 minutes actual time
    # code gets to 25 % at: 11:21 = 2 minutes actual time.

    global ichunk, pstart
    update_progress(float(ichunk / nchunks))
    ichunk += ncpu
    if ichunk == 4 * ncpu:
        ptime = nchunks * (time.time() - pstart) / 10 / 60
        print("Estimated time to completion: {:.2f} min".format(ptime))#, end='')
        # sys.stdout.flush()

    with pd.HDFStore(t1_file, 'r') as store:

        start = chunk_idx * chunksize
        stop = (chunk_idx + 1) * chunksize

        chunk = pd.read_hdf(
            t1_file, key, where="ievt >= {} & ievt < {}".format(start, stop))

        # if verbose:
        #     print("Chunk {}, start: {}  stop: {}, len: {},"
        #           .format(chunk_idx, start, stop, stop - start),
        #           "df shape:", chunk.shape)

    return intercom.process(chunk)
