#!/usr/bin/env python3
import time
import numpy as np
import pandas as pd
import pygama


def main():

    raw_file = "/Users/wisecg/dev/mj60/data/2018-10-9-P3LTP_Run42343"
    t1_file = "/Users/wisecg/dev/mj60/data/t1_run42343.h5"

    # tier0(raw_file)
    tier1(t1_file)


def tier0(raw_file):

    n_evt = 10000 # 487500 or np.inf

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
                      wf_names = ["waveform"], # can apply wfs to different calculators
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

    # do we want to write an object that can easily read wf_df?
    wfs = pygama.WaveformFrame(wf_df) # cool name bro



if __name__=="__main__":
    main()