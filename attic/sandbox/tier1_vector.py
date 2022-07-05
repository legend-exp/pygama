import os, time, re, h5py
import pandas as pd
import multiprocessing as mp
from functools import partial

from .vector import *
from ..decoders.io_base import *
from ..decoders.digitizers import *
from ..utils import *

def RunDSPVec(t1_file,
                    vec_process=None,
                    digitizer_list=None,
                    out_prefix="t2",
                    verbose=False,
                    output_dir=None,
                    multiprocess=False):
    """ vector version of tier 1 processor """

    if vec_process is None:
        vec_process = VectorProcess(default_list=True)

    print("Starting pygama Tier 1 (vector) processing ...")
    print("   Input file: {}".format(t1_file))
    statinfo = os.stat(t1_file)
    print("   File size: {}".format(sizeof_fmt(statinfo.st_size)))

    start = time.clock()
    directory = os.path.dirname(t1_file)
    output_dir = os.getcwd() if output_dir is None else output_dir

    # snag the run number (assuming t1_file ends in _run<number>.<filetype>)
    run_str = re.findall('run\d+', t1_file)[-1]
    run = int(''.join(filter(str.isdigit, run_str)))

    # get pygama's available digitizers
    if digitizer_list is None:
        digitizer_list = get_digitizers()

    # get digitizers in the file
    f = h5py.File(t1_file, 'r')
    digitizer_list = [d for d in digitizer_list if d.decoder_name in f.keys()]

    print("   Found digitizers:")
    for d in digitizer_list:
        print("   -- {}".format(d.decoder_name))

    for d in digitizer_list:
        print("Processing data from: " + d.decoder_name)

        object_info = pd.read_hdf(t1_file, key=d.class_name)
        d.load_object_info(object_info)

        # single thread process -- let's ABANDON THIS
        # t1_df = pd.read_hdf(t1_file, key=d.decoder_name)
        # t2_df = vec_process.Process(t1_df)

        # multi process -- i want to ALWAYS do this, using hdf5 chunking
        # even if i only have one thread available.
        # try to write each chunk to the file so you never hold the whole
        # file in memory.
        h5key = d.class_name
        chunksize = 3000 # num wf rows.  optimal for my mac, at least
        n_cpu = mp.cpu_count()

        with pd.HDFStore(t1_file, 'r') as store:
            nrows = store.get_storer(h5key).shape[0] # fixed only
            chunk_idxs = list(range(nrows//chunksize + 1))

        keywords = {"t1_file":t1_file, "chunksize":chunksize, "h5key":h5key}

        with mp.Pool(n_cpu) as p:
            result_list = p.map(partial(process_chunk, **keywords), chunk_idxs)

        # t2_df = pd.concat(result_list)
        #
        # print("Elapsed: {:.2f} sec".format(time.time()-t_start))


    t2_file = os.path.join(output_dir, "{}_run{}.h5".format(out_prefix, run))

    if verbose:
        print("Writing Tier 2 File:\n   {}".format(t2_file))
        print("   Entries: {}".format(len(t2_df)))
        print("   Data columns:")
        for col in t2_df.columns:
            print("   -- " + col)

    t2_df.to_hdf(
        t2_file,
        key="data",
        format='table',
        mode='w',
        data_columns=t2_df.columns.tolist())

    if verbose:
        statinfo = os.stat(t2_file)
        print("File size: {}".format(sizeof_fmt(statinfo.st_size)))
        elapsed = time.clock() - start
        proc_rate = elapsed/len(t2_df)
        print("Time elapsed: {:.2f} sec  ({:.5f} sec/wf)".format(elapsed, proc_rate))
        print("Done.")


def process_chunk(chunk_idx, t1_file, chunksize, h5key):

    # print("Processing chunk #{}".format(chunk_idx))

    with pd.HDFStore(t1_file, 'r') as store:
        start = chunk_idx * chunksize
        stop = (chunk_idx + 1) * chunksize
        # print("start: {}  stop: {}".format(start, stop))
        chunk = pd.read_hdf(t1_file, h5key,
                            where='index >= {} & index < {}'.format(start, stop))

    pyg = pygama.VectorProcess(default_list=True)

    t1_df = pyg.Process(chunk)

    return t1_df
