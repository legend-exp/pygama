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

from ..io.io_base import *
from ..io.digitizers import *
from ..utils import *


def RunDSP(t1_file, intercom, run=None, digitizers=None, ftype="default",
           output_dir=None, output_prefix="t2", overwrite=True, verbose=False,
           nevt=None, ioff=None, multiprocess=True, chunk=3000):

    t_start = time.time()

    if verbose:
        print("Starting pygama Tier 1 processing ...")
        print("  Date:", datetime.datetime.now())
        print("  Input file:", t1_file)

    if "~" in t1_file:
        t1_file = os.path.expanduser("~") + t1_file.split("~")[-1]

    if verbose:
        print("  Size: ", sizeof_fmt(os.path.getsize(t1_file)))


    # multiprocessing parameters
    CHUNKSIZE = chunk
    ncpu = mp.cpu_count()

    start = time.time()
    in_dir = os.path.dirname(t1_file)
    output_dir = os.getcwd() if output_dir is None else output_dir

    if run is None:
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
        # if t2_file is None:
        t2_file = os.path.join(output_dir, "t2_run{}.h5".format(run))

    if os.path.isfile(t2_file):
        if overwrite:
            if verbose:
                print("Overwriting existing file...")
            os.remove(t2_file)
        else:
            if verbose:
                print("File already exists, continuing ...")
            return

    # get digitizers
    if digitizers is None:

        decoders = []
        for sub in DataTaker.__subclasses__():
            tmp = sub() # instantiate the class
            decoders.append(tmp)

        # shouldn't this extend the list instead of overwriting it?
        with pd.HDFStore(t1_file, 'r') as store:
            keys = [key[1:] for key in store.keys()]  # remove leading '/'
            digitizers = [d for d in decoders if d.decoder_name in keys]

    # go running
    for d in digitizers:
        if verbose:
            print("Processing data from digitizer: {}".format(d.decoder_name))

        # get some info about the objects in the file
        with pd.HDFStore(t1_file, 'r') as store:

            s = store.get_storer(d.decoder_name)

            # if d.class_name is not None:
            #     object_info = store.get(d.decoder_name)
            #     d.load_metadata(object_info)

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

            if verbose:
                print(f"Found {nrows} rows, splitting into {nchunks} chunks")

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

            # main routine: use multiprocessing
            with mp.Pool(ncpu) as p:
                result_list = p.map(partial(process_chunk, **keywords), chunk_idxs)

            # # debug: process chunks in series
            # for idx in chunk_idxs:
            #     process_chunk(idx, **keywords)
            #     exit()

            t2_df = pd.concat(result_list)

            # print("it worked", len(t2_df), len(result_list[0]))

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

                t1_df = pd.read_hdf(t1_file, key=d.decoder_name,
                                    where = f"ievt > {ioff} & ievt < {ioff+nevt}")
            else:
                print("WARNING: no event limit set (-n option)")
                print("read the whole df into memory?  are you sure? (y/n)")
                if input() == "y":
                    t1_df = pd.read_hdf(t1_file, key=d.decoder_name)
                else:
                    exit()

            t2_df = intercom.process(t1_df, verbose)

    if verbose:
        update_progress(1)

    # ---------------- write Tier 2 output ----------------

    if verbose:
        print("Writing Tier 2 File:\n   {}".format(t2_file))
        print("  Entries: {}".format(len(t2_df)))

    # set everything except the index as a searchable data column
    dcols = [col for col in t2_df.columns.tolist() if "index" not in col]
    if verbose:
        print("  Data columns:\n", t2_df.columns.values)
    # print(t2_df.dtypes)
    # print(t2_df.shape)

    t2_df.to_hdf(
        t2_file,
        key="data",
        mode='w',
        format='table',
        data_columns=dcols,
        complib="blosc:snappy",
        complevel=2
        )

    if verbose:
        statinfo = os.stat(t2_file)
        print("  File size: {}".format(sizeof_fmt(statinfo.st_size)))
        elapsed = time.time() - start
        proc_rate = elapsed / len(t2_df)
        print("  Date:", datetime.datetime.now())
        print("  Time elapsed: {:.2f} min  ({:.5f} sec/wf)".format(
            elapsed / 60, proc_rate))
        print("Done.\n")


def process_chunk(chunk_idx, t1_file, chunksize, nchunks, ncpu, key, intercom,
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
    if verbose:
        update_progress(float(ichunk / nchunks))

    ichunk += ncpu

    if ichunk == 4 * ncpu:
        ptime = nchunks * (time.time() - pstart) / 10 / 60
        print(f"Estimated time to completion: {ptime:.2f} min")#, end='')
        # sys.stdout.flush()

    with pd.HDFStore(t1_file, 'r') as store:

        start = chunk_idx * chunksize
        stop = (chunk_idx + 1) * chunksize

        # this was ievt before, this requires a reset_index before calling this.
        # it probably partially fixes the "pygama append bug"
        # chunk = pd.read_hdf(t1_file, key, where=f"index >= {start} & index < {stop}")

        # had to change this back to get MJ60 processing working.  at some point
        # I need to look into the difference between ievt and index
        chunk = pd.read_hdf(t1_file, key, where=f"ievt >= {start} & ievt < {stop}")
        # print("")
        # print(chunk)
        # exit()

        # super verbose
        # print("Chunk {}, start: {}  stop: {}, len: {},"
        #       .format(chunk_idx, start, stop, stop - start),
        #       "df shape:", chunk.shape)

    return intercom.process(chunk)
