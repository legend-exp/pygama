"""
pygama tier 1 processing
tier 1 data --> DSP --> tier 2 (i.e. gatified)
"""
import os, re, sys, time
import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial

from ..io.decoders.data_loading import *
from ..io.decoders.digitizers import *
from ..utils import *

def ProcessTier1(t1_file,
                 processor,
                 digitizers=None,
                 output_dir=None,
                 overwrite=True,
                 verbose=False,
                 nevt=None,
                 multiprocess=True,
                 chunk = 3000):

    print("Starting pygama Tier 1 processing ...")
    print("   Input file: {}".format(t1_file))
    t_start = time.time()

    # multiprocessing parameters
    CHUNKSIZE = chunk
    NCPU = mp.cpu_count()

    start = time.time()
    in_dir = os.path.dirname(t1_file)
    output_dir = os.getcwd() if output_dir is None else output_dir

    # snag the run number (assuming t1_file ends in _run<number>.<filetype>)
    run_str = re.findall('run\d+', t1_file)[-1]
    run = int(''.join(filter(str.isdigit, run_str)))

    # declare output file
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
            keys = [key[1:] for key in store.keys()] # remove leading '/'
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
            elif isinstance(s, pd.io.pytables.FrameFixed):
                use_pytables = False
            else:
                print("Unknown type!", type(s))
                exit()

        # --------------- run multiprocessing ----------------
        if use_pytables and multiprocess:

            print("Found {} rows".format(nrows))

            keywords = {"t1_file":t1_file,
                        "chunksize":CHUNKSIZE,
                        "nchunks":len(chunk_idxs),
                        "key":d.decoder_name,
                        "processor":processor,
                        "verbose":verbose}

            with mp.Pool(NCPU) as p:
                result_list = p.map(partial(process_chunk, **keywords),
                                    chunk_idxs)

            # debug: process chunks linearly
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
                t1_df = pd.read_hdf(t1_file, key=d.decoder_name,
                                    where="ievt < {}".format(nevt))
            else:
                t1_df = pd.read_hdf(t1_file, key=d.decoder_name)

            t2_df = processor.process(t1_df, verbose)

    update_progress(1)

    # ---------------- write Tier 2 output ----------------

    if verbose:
        print("Writing Tier 2 File:\n   {}".format(t2_file))
        print("   Entries: {}".format(len(t2_df)))
        print("   Data columns:")
        for col in t2_df.columns:
            print("   -- " + str(col))

    t2_df.to_hdf(
        t2_file,
        key="data",
        format='table',
        mode='w',
        data_columns=t2_df.columns.tolist())

    if verbose:
        statinfo = os.stat(t2_file)
        print("File size: {}".format(sizeof_fmt(statinfo.st_size)))
        elapsed = time.time() - start
        proc_rate = elapsed/len(t2_df)
        print("Time elapsed: {:.2f} sec  ({:.5f} sec/wf)".format(elapsed, proc_rate))
        print("Done.")


def process_chunk(chunk_idx, t1_file, chunksize, nchunks,
                  key, processor, verbose=False):
    """
    use hdf5 indexing, which is way faster than reading in the df first.
    this is a really good reason to use the 'tables' format
    """
    # this makes the progress bar jump around randomly, should fix someday w/ a counter
    update_progress(float(chunk_idx/nchunks))

    with pd.HDFStore(t1_file, 'r') as store:

        start = chunk_idx * chunksize
        stop = (chunk_idx + 1) * chunksize

        chunk = pd.read_hdf(t1_file, key,
                            where="ievt >= {} & ievt < {}"
                            .format(start, stop))
        if verbose:
            print("Chunk {}, start: {}  stop: {}, len: {},"
                  .format(chunk_idx, start, stop, stop-start),
                  "df shape:", chunk.shape)

    return processor.process(chunk)
