import pandas as pd
import sys
import time
import numpy as np
import scipy as sp
import scipy.optimize as opt
import scipy.signal as signal
import os, json
import pygama.dataset as ds
import pygama.analysis.histograms as pgh
import pygama.dsp.transforms as pgt
import pygama.utils as pgu
import pygama.analysis.peak_fitting as pga
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.style.use('style.mplstyle')


def main():

    if(len(sys.argv) != 2):
        print('Usage: wfs.py [run number]')
        sys.exit()

    plot_wfs()
    #flip_through_wfs()   

def plot_wfs():

    start = time.time()

    df = pd.read_hdf('~/Data/MJ60/pygama/t1_run'+sys.argv[1]+'.h5', '/ORSIS3302DecoderForEnergy')

    runtime = ds.DataSet(run=int(sys.argv[1]), md='./runDB.json').get_runtime()
    counts_per_second = (len(df))/runtime

    df = df.reset_index(drop=True)
    del df['energy']
    del df['channel']
    del df['energy_first']
    del df['ievt']
    del df['packet_id']
    del df['timestamp']
    del df['ts_hi']
    del df['ts_lo']

    def bl_sub(wf):
        return df.loc[wf,:]-df.iloc[wf,0:850].mean()
        
    nsamp = 3000
 
    xvals = np.arange(0,nsamp)

    for i in range(6,12):
        plt.plot(xvals, bl_sub(i), lw=1)
    plt.xlabel('Sample Number', ha='right', x=1.0)
    plt.ylabel('ADC Value', ha='right', y=1.0)
    plt.tight_layout()
    print('total runtime = {} seconds'.format(runtime))
    print('counting rate = {} counts/second'.format(counts_per_second))
    print('python script time = {} seconds'.format(time.time() - start))
    plt.show()

def flip_through_wfs():
    
    df = pd.read_hdf('~/Data/MJ60/pygama/t1_run'+sys.argv[1]+'.h5', '/ORSIS3302DecoderForEnergy')
    df = df.reset_index(drop=True)
    del df['energy']
    del df['channel']
    del df['energy_first']
    del df['ievt']
    del df['packet_id']
    del df['timestamp']
    del df['ts_hi']
    del df['ts_lo']

    def bl_sub(wf):
        return df.loc[wf,:]-df.iloc[wf,0:850].mean()

    def savgol(wf):
        return signal.savgol_filter(df.loc[wf,:]-df.iloc[wf,0:850].mean(), 47, 2)

    def trap(wf, rise, flat, fall=None, decay=0):
        nsamp = 3000
        rt, ft, dt = int(rise * nsamp), int(flat * nsamp), decay * nsamp
        flt = rt if fall is None else int(fall * nsamp)
        return rise

    nsamp = 3000

    xvals = np.arange(0,nsamp)

    i = -1
    while True:
        if i != -1:
            inp = input()
            if inp == "q": exit()
            if inp == "p": i -= 2
        i += 1
        print(i)

        plt.cla()
        plt.plot(xvals, bl_sub(i), color="black", lw=2, label="raw wf, run {}, wf {}".format(str(sys.argv[1]),i))
        plt.plot(xvals, savgol(i), color="red", lw=1, label="Savitzky-Golay Filter")
        plt.xlabel('Sample Number', ha='right', x=1.0)
        plt.ylabel('ADC Value', ha='right', y=1.0)
        plt.legend(frameon=True, loc='upper left', fontsize='small')
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.01)

if __name__ == '__main__':
        main()

