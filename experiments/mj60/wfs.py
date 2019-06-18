#!/usr/bin/env python3
import pandas as pd
import sys
import time
import numpy as np
import scipy as sp
import scipy.optimize as opt
from pygama import DataSet
import pygama.analysis.histograms as pgh
import pygama.utils as pgu
import pygama.analysis.peak_fitting as pga
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# plt.style.use('style.mplstyle')


def main():

    if(len(sys.argv) != 2):
        print('Usage: wfs.py [run number]')
        sys.exit()

    plot_wf()

def plot_wf():

    start = time.time()

    file = "~/Data/MJ60/pygama/t1_run"+sys.argv[1]+'.h5', "/ORSIS3302DecoderForEnergy"
    file = os.path.expanduser(file)
    # if file contains $ENVVAR : use os.path.expandvars(file)
    df = pd.read_hdf(file)
    print(len(df))

    df = df.reset_index(drop=True)
    del df['energy']
    del df['channel']
    del df['energy_first']
    del df['ievt']
    del df['packet_id']
    del df['timestamp']
    del df['ts_hi']
    del df['ts_lo']

    #c = df.iloc[0,0:2].mean()
    #print(c)
    #exit()
    xvals = np.arange(0,3000)

    for i in range(6,12):
        plt.plot(xvals, df.loc[i,:]-df.iloc[i,0:850].mean(), lw=1)
    plt.xlabel('Sample Number', ha='right', x=1.0)
    plt.ylabel('ADC Value', ha='right', y=1.0)
    plt.tight_layout()
    print('Time = {} seconds'.format(time.time() - start))
    plt.show()

############
#Clints method for only a block of wfs

# t1_file = os.path.expandvars("~/Data/MJ60/pygama/t1_run204.h5")
#     # with pd.HDFStore(t1_file, 'r') as store:
#     #     print(store.keys())
#
#     key = "/ORSIS3302DecoderForEnergy"
#     chunk = pd.read_hdf(t1_file, key, where="ievt < {}".format(1000))
#     chunk.reset_index(inplace=True) # required step -- fix pygama "append" bug
#
#     # create waveform block.  mask wfs of unequal lengths
#     icols = []
#     for idx, col in enumerate(chunk.columns):
#         if isinstance(col, int):
#             icols.append(col)
#     wf_block = chunk[icols].values
#     print(wf_block.shape, type(wf_block))

if __name__ == '__main__':
        main()
