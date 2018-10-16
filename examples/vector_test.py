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
                      wf_names = ["waveform"], fun_args = {"i_end":700})

    pyg.AddTransformer(pygama.processing.vectorized.bl_subtract,
                       wf_names = ["waveform"],
                       fun_args = {})

    # "gatified" only
    t1_df = pyg.Process(event_df)

    # include waveforms
    # t1_df, t1wf_df = pyg.Process(event_df, ["waveform"])

    # print some results
    # print(t1_df.shape, t1_df.columns)
    # print(t1_df[["bl_avg","bl_int","bl_slope"]])




if __name__=="__main__":
    main()