#!/usr/bin/env python3
import time
import numpy as np
import pandas as pd
import pygama


def main():

    raw_file = "/Users/wisecg/dev/mj60/data/2018-10-9-P3LTP_Run42343"
    t1_file = "/Users/wisecg/dev/mj60/data/t1_run42343.h5"

    # tier0(raw_file)
    # tier1(t1_file)
    # tier1_quick(t1_file)
    tier1_mp(t1_file)


def tier0(raw_file):

    # n_evt = 100000 # 487500 or np.inf
    n_evt = 5000

    from pygama.processing._tier0 import ProcessTier0
    ProcessTier0(raw_file,
                 verbose=True,
                 output_dir="/Users/wisecg/dev/mj60/data",
                 n_max=n_evt,
                 chan_list=None)

def tier1(t1_file):

    digitizer = pygama.decoders.digitizers.Gretina4MDecoder(
        correct_presum = False,
        split_waveform = False,
        )

    event_df = pd.read_hdf(t1_file, key = digitizer.decoder_name)

    pyg = pygama.VectorProcess()
    # pyg = pygama.ScalarProcess() # just an idea

    pyg.AddCalculator(pygama.processing.vectorized.avg_baseline,
                      wf_names = ["waveform"],
                      fun_args = {"i_end":700})

    pyg.AddCalculator(pygama.processing.vectorized.fit_baseline,
                      wf_names = ["waveform"],
                      fun_args = {"i_end":700})

    pyg.AddTransformer(pygama.processing.vectorized.bl_subtract,
                       wf_names = ["waveform"],
                       fun_args = {"test":False})

    # process as "gatified" only
    # t1_df = pyg.Process(event_df)
    # print(t1_df.shape, t1_df.columns)
    # print(t1_df[["bl_avg","bl_int","bl_slope"]])

    # process as "gatified" and output waveforms
    t1_df, wf_df = pyg.Process(event_df, ["waveform", "wf_blsub"])
    # print(type(wf_df), wf_df.shape, wf_df.columns)
    # print(wf_df["waveform"][0])
    # print(wf_df["wf_blsub"][0])

    # write an object that can easily read wf_df
    wfs = pygama.WaveformFrame(wf_df)


def tier1_quick(t1_file):

    event_df = pd.read_hdf(t1_file, "ORGretina4MWaveformDecoder")
    pyg = pygama.VectorProcess(default_list=True)
    t1_df = pyg.Process(event_df)
    print(t1_df.to_string())


def tier1_mp(t1_file):

    import multiprocessing as mp
    from functools import partial

    h5key = "ORGretina4MWaveformDecoder"
    chunksize = 40

    with pd.HDFStore(t1_file, 'r') as store:
        nrows = store.get_storer(h5key).nrows
        chunk_idxs = list(range(nrows//chunksize + 1))
    chunk_idxs = [0] # debug

    keywords = {"t1_file":t1_file, "chunksize":chunksize, "h5key":h5key}

    with mp.Pool(mp.cpu_count()) as p:
        result_list = p.map(partial(process_chunk, **keywords), chunk_idxs)

    # df_final = pd.concat(result_list)
    # print(df_final.to_string())


def process_chunk(chunk_idx, t1_file, chunksize, h5key):

    with pd.HDFStore(t1_file, 'r') as store:

        # chunk = pd.read_hdf(t1_file, h5key)

        start = chunk_idx * chunksize
        stop = (chunk_idx + 1) * chunksize
        chunk = pd.read_hdf(t1_file, h5key,
                            where='index >= {} & index <= {}'.format(start, stop))

    pyg = pygama.VectorProcess(default_list=True)
    t1_df, wf_df = pyg.Process(chunk, ['waveform'])

    print(t1_df)
    # exit()
    return t1_df



if __name__=="__main__":
    main()