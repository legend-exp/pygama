#!/usr/bin/env python3
import time, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_dir = "/Users/wisecg/project/pygama"

# run = 42343 # mjd data
run = 72 # mj60 data

# user build options
# could read these in from json config files if they got long
t0_options = {
    42343: {"digitizer":"ORGretina4MWaveformDecoder", "n_blsamp":500},
    72: {"digitizer":"ORSIS3302DecoderForEnergy",
         "window":"max", # max or tp
         "n_samp":2000,
         "n_blsamp":10000}
    }
t1_options = {
    42343 : {"fit_baseline": {"i_end":500}},
    72 : {"fit_baseline": {"i_end":800}}
    }


def main():

    # ------- TIER 0 -------

    # tier0(run, n_evt=50000) # np.inf by default
    # tier0_check(run)

    # ------- TIER 1 -------

    # tier1(run)

    # ------- TIER 2 -------

    tier2(run) # display Tier 2 data and maybe run analysis

    # ----------------------


def tier0(run, n_evt=None):

    from pygama.processing.tier0 import ProcessTier0

    raw_file = glob.glob("{}/*Run{}".format(data_dir, run))[0]

    if n_evt is None:
        n_evt = np.inf

    ProcessTier0(raw_file,
                 verbose=True,
                 output_dir=data_dir,
                 n_max=n_evt,
                 settings=t0_options[run])


def tier0_check(run):

    t1_file = glob.glob("{}/t1_run{}.h5".format(data_dir, run))[0]

    with pd.HDFStore(t1_file,'r') as store:
        print("keys found:", store.keys())
        print("INFO:\n", store.info())

        key = "/"+t0_options[run]["digitizer"]

        # t1_df = store.get(key)
        # print(t1_df.shape)

        # preamp_df = store.get("/ORMJDPreAmpDecoderForAdc")
        # print(preamp_df.shape)
        # print(preamp_df)

        nrows = store.get_storer(key).nrows # tables only
        # nrows = store.get_storer(h5keys[run]).shape[0] # fixed only

        print("found {} rows".format(nrows))


def tier1(run):

    from pygama.processing.base import Tier1Processor
    from pygama.processing.tier1 import ProcessTier1

    t1_file = glob.glob("{}/t1_run{}.h5".format(data_dir, run))[0]

    # proc = Tier1Processor(t1_options, default_list=True) # cop out

    proc = Tier1Processor(t1_options)
    proc.add("fit_baseline", {"i_end":800})
    proc.add("bl_subtract")
    proc.add("trap_filter")
    proc.add("trap_max", {"test":False})

    # NOTE: using {'test':True} displays a plot,
    # won't work when multiprocessing is enabled

    ProcessTier1(t1_file,
                 proc,
                 out_prefix="t2",
                 out_dir=data_dir,
                 verbose=True,
                 multiprocess=True,
                 settings=t1_options[run])


def tier2(run):

    plt.style.use("./clint.mpl")
    from pygama.utils import get_hist

    t2_file = glob.glob("{}/t2_run{}.h5".format(data_dir, run))[0]
    t2_df = pd.read_hdf(t2_file)

    x_lo, x_hi, xpb = 0, 8000, 10
    nb = int((x_hi-x_lo)/xpb)

    # we can do this but i don't like its look
    # t2_df.hist("trap_max", bins=nb, range=(x_lo, x_hi))

    # def get_hist(np_arr, x_lo, x_hi, xpb, nb=None, shift=True, wts=None):

    xH, yH = get_hist(t2_df["trap_max"], x_lo, x_hi, xpb)

    plt.plot(xH, yH, c='b', lw=1, ls='steps', label="trap_max")
    plt.xlabel("Energy [uncal]", ha='right', x=1)
    plt.ylabel("Counts [arb]", ha='right', y=1)
    plt.legend()
    plt.tight_layout()
    plt.show()


    # plt.show()


if __name__=="__main__":
    main()