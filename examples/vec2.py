#!/usr/bin/env python3
import time, glob
import numpy as np
import pandas as pd
import pygama

from pygama.processing.tier0 import ProcessTier0
from pygama.processing.vector import *
from pygama.processing.tier1_vector import ProcessTier1Vec

data_dir = "/Users/wisecg/project/pygama"

# run = 42343 # mjd data
run = 72 # mj60 data

h5keys = {42343:"/ORGretina4MWaveformDecoder",
          72:"/ORSIS3302DecoderForEnergy"}

def main():
    """ I need to get a Tier 1 format that is:
        - chunkable (paralellizable)
        - embeds waveforms in a cell
        - works for both Gretina and SIS3302 cards
    PyTables can't store more than ~2500 columns, so a "table" option is not
    desirable for the Tier 1 output, but it IS desirable for the Tier 2 output.
    """
    # ------- TIER 0 -------

    # tier0(run, n_evt=1000)
    # check_tier0(run)

    # ------- TIER 1 -------

    tier1vec(run) # examples of utilizing both types of processors
    # tier1base(run) # could even combine the resulting dataframes

    # ------- TIER 2 -------

    # tier2(run) # display Tier 2 data and maybe run analysis

    # ----------------------


def tier0(run, n_evt=None):

    raw_file = glob.glob("{}/*Run{}".format(data_dir, run))[0]

    if n_evt is None:
        n_evt = np.inf

    ProcessTier0(raw_file,
                 verbose=True,
                 output_dir=data_dir,
                 n_max=n_evt)


def check_tier0(run):

    t1_file = glob.glob("{}/t1_run{}.h5".format(data_dir, run))[0]

    with pd.HDFStore(t1_file,'r') as store:
        print("keys found:", store.keys())
        # nrows = store.get_storer(h5keys[run]).nrows # tables only
        nrows = store.get_storer(h5keys[run]).shape[0] # fixed only
    print("nrows:", nrows)


def tier1vec(run):

    t1_file = glob.glob("{}/t1_run{}.h5".format(data_dir, run))[0]

    # set different options for sis3302 and gretina
    i_end = {72:10000, 42343:500}

    # declare a "gatified" list of calculations
    vec_process = VectorProcess()
    vec_process.AddCalculator(fit_baseline, fun_args = {"i_end":i_end[run]})
    vec_process.AddTransformer(bl_subtract, fun_args = {"test":False})
    vec_process.AddTransformer(trap_filter, fun_args = {"test":False})

    ProcessTier1Vec(t1_file,
                    vec_process,
                    out_prefix="t2",
                    verbose=True,
                    multiprocess=False)



if __name__=="__main__":
    main()