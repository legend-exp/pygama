#!/usr/bin/env python3
import time, glob
import numpy as np
import pandas as pd

data_dir = "/Users/wisecg/project/pygama"

run = 42343 # mjd data
# run = 72 # mj60 data

# user options
t0_options = {
    42343: {"digitizer":"ORGretina4MWaveformDecoder", "n_blsamp":500},
    72: {"digitizer":"ORSIS3302DecoderForEnergy",
         "window":"max", # max or tp
         "n_samp":2000,
         "n_blsamp":10000}
    }
t1_options = {
    43243 : {"fit_baseline": {"i_end":500}},
    72 : {"fit_baseline": {"i_end":8000}}
    }

def main():
    """
    ProcessRaw needs to output DAQ data in a format that is:
    - parallelizable, requires hdf5 tables
    - chunkable (allows vectorizing calculators in RunDSP)
    - doesn't embed waveforms in a cell (tables output doesn't work for 'fixed')
    - wf tables don't exceed pytables dimension limits
    - works for both Gretina and SIS3302 cards (shows versatility)
    """
    # ------- TIER 0 -------

    daq_to_raw(run, n_evt=np.inf)
    # check_daq_to_raw(run)

    # ------- TIER 1 -------

    raw_to_dsp(run)

    # ------- TIER 2 -------

    # tier2(run) # display Tier 2 data and maybe run analysis

    # ----------------------


def daq_to_raw(run, n_evt=None):

    from pygama.processing.daq_to_raw import ProcessRaw

    raw_file = glob.glob("{}/*Run{}".format(data_dir, run))[0]

    if n_evt is None:
        n_evt = np.inf

    ProcessRaw(raw_file,
                 verbose=True,
                 output_dir=data_dir,
                 n_max=n_evt,
                 settings=t0_options[run])


def check_daq_to_raw(run):

    t1_file = glob.glob("{}/t1_run{}.h5".format(data_dir, run))[0]

    with pd.HDFStore(t1_file,'r') as store:
        print("keys found:", store.keys())
        print("INFO:\n", store.info())

        t1_df = store.get("/ORGretina4MWaveformDecoder")
        print(t1_df.shape)

        # preamp_df = store.get("/ORMJDPreAmpDecoderForAdc")
        # print(preamp_df.shape)
        # print(preamp_df)

        # nrows = store.get_storer(h5keys[run]).nrows # tables only
        # nrows = store.get_storer(h5keys[run]).shape[0] # fixed only


def raw_to_dsp(run):

    from pygama.processing.dsp_base import Tier1Processor
    from pygama.processing.raw_to_dsp import RunDSP

    t1_file = glob.glob("{}/t1_run{}.h5".format(data_dir, run))[0]

    # proc = Tier1Processor(default_list=True)
    proc = Tier1Processor()
    proc.AddCalculator(fit_baseline, fun_args = {"i_end":5000})
    proc.AddTransformer(bl_subtract, fun_args = {"test":False})
    proc.AddTransformer(trap_filter, fun_args = {"test":False})

    RunDSP(t1_file,
                 proc,
                 out_prefix="t2",
                 verbose=True,
                 multiprocess=True)


if __name__=="__main__":
    main()