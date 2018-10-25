#!/usr/bin/env python3
import time, glob
import numpy as np
import pandas as pd
import pygama

data_dir = "/Users/wisecg/project/pygama"

h5keys = {42343:"/ORGretina4MWaveformDecoder",
          72:"ORSIS3303DecoderForEnergy"}

def main():
    """ I need to get a Tier 1 format that is:
        - chunkable (paralellizable)
        - embeds waveforms in a cell
        - works for both Gretina and SIS3302 cards
    PyTables can't store more than ~2500 columns, so a "table" option is not
    desirable for the Tier 1 output, but it IS desirable for the Tier 2 output.
    """
    run = 42343
    # run = 72

    # tier1(run, n_evt=1000)
    check_tier1(run)
    # tier2(run)


def tier1(run, n_evt=None):

    raw_file = glob.glob("{}/*Run{}".format(data_dir, run))[0]

    if n_evt is None:
        n_evt = np.inf

    from pygama.processing.tier0 import ProcessTier0
    ProcessTier0(raw_file,
                 verbose=True,
                 output_dir=data_dir,
                 n_max=n_evt)


def check_tier1(run):

    t1_file = glob.glob("{}/t1_run{}.h5".format(data_dir, run))[0]

    with pd.HDFStore(t1_file, 'r') as store:

        nrows = store.get_storer(h5keys[run]).nrows
        # chunk_idxs = list(range(nrows//chunksize + 1))

    print(nrows)




if __name__=="__main__":
    main()