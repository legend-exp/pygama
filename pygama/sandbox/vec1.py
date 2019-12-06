#!/usr/bin/env python3
import time
import numpy as np
import pandas as pd
import pygama


def main():

    raw_file = "/Users/wisecg/dev/mj60/data/2018-10-9-P3LTP_Run42343"
    t1_file = "/Users/wisecg/dev/mj60/data/t1_run42343.h5"

    # daq_to_raw(raw_file)
    # raw_to_dsp(t1_file)
    raw_to_dsp_quick(t1_file)
    # raw_to_dsp_mp(t1_file)
    # check_equal(t1_file)
    # optimize_raw_to_dsp_mp(t1_file)


def daq_to_raw(raw_file, n_evt=None):

    if n_evt is None:
        n_evt = 10000 # 487500 or np.inf
        # n_evt = 5000

    from pygama.processing.daq_to_raw import ProcessRaw
    ProcessRaw(raw_file,
                 verbose=True,
                 output_dir="/Users/wisecg/dev/mj60/data",
                 n_max=n_evt,
                 chan_list=None)


def raw_to_dsp(t1_file):

    digitizer = pygama.decoders.digitizers.Gretina4M(
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

    # TODO: write an object that can easily read wf_df
    wfs = pygama.WaveformFrame(wf_df)


def raw_to_dsp_quick(t1_file):

    t_start = time.time()

    # event_df = pd.read_hdf(t1_file, "ORGretina4MWaveformDecoder")#, stop=40)
    # print(event_df)
    # exit()

    n_evt = 100
    # daq_to_raw("/Users/wisecg/dev/mj60/data/2018-10-9-P3LTP_Run42343", n_evt)

    # use index (can also use anything in 'data_columns')
    # event_df = pd.read_hdf(t1_file, "ORGretina4MWaveformDecoder",
                           # where='index >= {} & index < {}'.format(15, 30))#, stop=40)

    event_df = pd.read_hdf(t1_file, "ORGretina4MWaveformDecoder")

    pyg = pygama.VectorProcess(default_list=True)
    t1_df = pyg.Process(event_df)

    print("done. final shape:", t1_df.shape)
    # print(t1_df.to_string())
    print("Elapsed: {:.2f} sec".format(time.time()-t_start))

    return t1_df


def raw_to_dsp_mp(t1_file):

    import multiprocessing as mp
    from functools import partial

    n_evt = np.inf
    # daq_to_raw("/Users/wisecg/dev/mj60/data/2018-10-9-P3LTP_Run42343", n_evt)

    t_start = time.time()

    h5key = "ORGretina4MWaveformDecoder"
    chunksize = 3000 # in rows.  optimal for my mac, at least
    n_cpu = mp.cpu_count()

    with pd.HDFStore(t1_file, 'r') as store:
        nrows = store.get_storer(h5key).nrows
        chunk_idxs = list(range(nrows//chunksize + 1))

    keywords = {"t1_file":t1_file, "chunksize":chunksize, "h5key":h5key}

    with mp.Pool(n_cpu) as p:
        result_list = p.map(partial(process_chunk, **keywords), chunk_idxs)

    mp_df = pd.concat(result_list)

    print("Elapsed: {:.2f} sec".format(time.time()-t_start))

    return mp_df


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


def check_equal(t1_file):

    full_df = raw_to_dsp_quick(t1_file)
    mp_df = raw_to_dsp_mp(t1_file)

    print("mp df:",mp_df.shape, "standard df:",full_df.shape)

    # compare the two dataframes, make sure they're equal
    # since we have floating point stuff, df1.equals(df2) won't always work
    # so compare row by row and don't worry too much about precision

    from collections import Counter
    for i in range(len(mp_df)):
        mpdf = ["{:.10e}".format(v) for v in mp_df.iloc[i].values]
        fulldf = ["{:.10e}".format(v) for v in full_df.iloc[i].values]
        if Counter(mpdf) != Counter(fulldf):
            print("DAMN.\n  MP:",mpdf,"\n  FULL:",fulldf)
        # else:
            # print(i, "YEAH")


def optimize_raw_to_dsp_mp(t1_file):
    """ seems that the sweet spot on my mac is chunksize ~ 3000 """

    import multiprocessing as mp
    from functools import partial

    n_evt = 200000
    # daq_to_raw("/Users/wisecg/dev/mj60/data/2018-10-9-P3LTP_Run42343", n_evt)
    h5key = "ORGretina4MWaveformDecoder"
    n_cpu = mp.cpu_count()
    n_cpu = 2
    print("CPUs used:",n_cpu)

    # it's pretty clear the sweet spot is between 500 and 5000
    # {500: 12.53, 1000: 9.52, 5000: 11.24, 10000: 12.09, 50000: 105.37}

    # n_chunk = [500, 1000, 5000, 10000, 50000] # my mac hates more than 50k
    # n_chunk = np.logspace(3, 4, num=20).astype(int) # 100 and 5000
    n_chunk = [1000, 3000, 6000]
    # print(n_chunk)
    # exit()

    t_chunk = []

    for chunksize in n_chunk:

        t_start = time.time()

        with pd.HDFStore(t1_file, 'r') as store:
            nrows = store.get_storer(h5key).nrows # this doesn't work for 'fixed'
            # nrows = store.get_storer("ORSIS3302DecoderForEnergy").shape[0]
            chunk_idxs = list(range(nrows//chunksize + 1))

        keywords = {"t1_file":t1_file, "chunksize":chunksize, "h5key":h5key}

        with mp.Pool(mp.cpu_count()) as p:
            result_list = p.map(partial(process_chunk, **keywords), chunk_idxs)

        mp_df = pd.concat(result_list)

        elap = time.time() - t_start
        print("chunk size:", chunksize, "elapsed:", elap)

        t_chunk.append(elap)

    print(dict(zip(n_chunk, t_chunk)))


if __name__=="__main__":
    main()