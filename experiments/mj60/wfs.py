#!/usr/bin/env python3
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
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
plt.style.use('style.mplstyle')

# for this code to work you need the tier1, tier2, and spectrum files for the run of interest.

def main():

    # plot_wfs()
    flip_through_wfs()
    #ADC_difference()
    #ADC_difference_cut()

def plot_wfs():

    if(len(sys.argv) != 2):
        print('Usage: wfs.py [run number]')
        sys.exit()

    start = time.time()

    with open("runDB.json") as f:
        runDB = json.load(f)
    tier_dir = os.path.expandvars(runDB["tier_dir"])

    df = pd.read_hdf('{}/t1_run{}.h5'.format(tier_dir,sys.argv[1]), '/ORSIS3302DecoderForEnergy')

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

    for i in range(0,1):
        plt.plot(xvals, bl_sub(i), lw=1)
    plt.xlabel('Sample Number', ha='right', x=1.0)
    plt.ylabel('ADC Value', ha='right', y=1.0)
    plt.tight_layout()
    print('total runtime = {} seconds'.format(runtime))
    print('counting rate = {} counts/second'.format(counts_per_second))
    print('python script time = {} seconds'.format(time.time() - start))
    plt.show()

def flip_through_wfs():

    if(len(sys.argv) != 4):
        print('Usage: wfs.py [run number] [lower energy limit (keV)] [upper energy limit (keV)]')
        sys.exit()

    with open("runDB.json") as f:
        runDB = json.load(f)
    tier_dir = os.path.expandvars(runDB["tier_dir"])
    meta_dir = os.path.expandvars(runDB["meta_dir"])

    df = pd.read_hdf('{}/t1_run{}.h5'.format(tier_dir,sys.argv[1]), '/ORSIS3302DecoderForEnergy')
    #df_2 = pd.read_hdf("{}/Spectrum_{}.hdf5".format(meta_dir,sys.argv[1]), key="df")

    #df_2 = df_2.reset_index(drop=True)
    df = df.reset_index(drop=True)
    del df['energy']
    del df['channel']
    del df['energy_first']
    del df['ievt']
    del df['packet_id']
    del df['timestamp']
    del df['ts_hi']
    del df['ts_lo']

    #df['e_cal'] = df_2['e_cal']
    #df = df.loc[(df.e_cal>int(sys.argv[2]))&(df.e_cal<int(sys.argv[3]))]
    #df = df.reset_index(drop=True)
    #df_3 = pd.DataFrame(df['e_cal'])
    #del df['e_cal']

    def bl_sub(wf):
        return df.loc[wf,:]-df.iloc[wf,0:500].mean()

    def savgol(wf):
        return signal.savgol_filter(df.loc[wf,:]-df.iloc[wf,0:500].mean(), 47, 2)

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
        plt.plot(xvals, bl_sub(i), color="black", lw=2, label="raw wf, run {}".format(str(sys.argv[1])))
        plt.plot(xvals, savgol(i), color="red", lw=1, label="Savitzky-Golay Filter")
        plt.xlabel('Sample Number', ha='right', x=1.0)
        plt.ylabel('ADC Value', ha='right', y=1.0)
        plt.legend(frameon=True, loc='upper left', fontsize='x-small')
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.01)

def ADC_difference():

    if(len(sys.argv) != 2):
        print('Usage: wfs.py [run number]')
        sys.exit()

    start = time.time()

    with open("runDB.json") as f:
        runDB = json.load(f)
    tier_dir = os.path.expandvars(runDB["tier_dir"])
    meta_dir = os.path.expandvars(runDB["meta_dir"])

    df = pd.read_hdf('{}/t1_run{}.h5'.format(tier_dir,sys.argv[1]), '/ORSIS3302DecoderForEnergy')
    df_2 = pd.read_hdf("{}/Spectrum_{}.hdf5".format(meta_dir,sys.argv[1]), key="df")

    df_2 = df_2.reset_index(drop=True)
    df = df.reset_index(drop=True)
    del df['energy']
    del df['channel']
    del df['energy_first']
    del df['ievt']
    del df['packet_id']
    del df['timestamp']
    del df['ts_hi']
    del df['ts_lo']
    #df_2['diff'] = [0]*len(df_2)

    #def dADC(wf):
        #return df.iloc[wf,1499:3000].mean()-df.iloc[wf,0:500].mean()

    #for i in range(len(df)):
        #df_2['diff'][i] = dADC(i)

    df_2['diff'] = df.iloc[:,1499:3000].mean(axis=1) - df.iloc[:,0:500].mean(axis=1)
    df_2['ratio'] = df_2['diff']/df_2['e_cal']
    print('python script time = {:.0f} seconds'.format(time.time() - start))

    plt.hist2d(df_2['e_cal'], df_2['diff'], np.arange(-5,100,0.1), norm=LogNorm())
    plt.xlim(0,50)
    plt.ylim(-5,100)
    plt.xlabel('Energy (keV)', ha='right', x=1.0)
    plt.ylabel('dADC/E', ha='right', y=1.0)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts')
    plt.tight_layout()
    plt.show()

def ADC_difference_cut():

    if(len(sys.argv) != 2):
        print('Usage: wfs.py [run number]')
        sys.exit()

    start = time.time()

    with open("runDB.json") as f:
        runDB = json.load(f)
    tier_dir = os.path.expandvars(runDB["tier_dir"])
    meta_dir = os.path.expandvars(runDB["meta_dir"])

    df = pd.read_hdf('{}/t1_run{}.h5'.format(tier_dir,sys.argv[1]), '/ORSIS3302DecoderForEnergy')
    df_2 = pd.read_hdf("{}/Spectrum_{}.hdf5".format(meta_dir,sys.argv[1]), key="df")

    df_2 = df_2.reset_index(drop=True)
    df = df.reset_index(drop=True)
    del df['energy']
    del df['channel']
    del df['energy_first']
    del df['ievt']
    del df['packet_id']
    del df['timestamp']
    del df['ts_hi']
    del df['ts_lo']
    #df_2['diff'] = [0]*len(df_2)

    #def dADC(wf):
        #return df.iloc[wf,1499:3000].mean()-df.iloc[wf,0:500].mean()

    #for i in range(len(df)):
        #df_2['diff'][i] = dADC(i)

    df_2['diff'] = df.iloc[:,1499:3000].mean(axis=1) - df.iloc[:,0:500].mean(axis=1)
    df_2['ratio'] = df_2['diff']/df_2['e_cal']

    df_2 = df_2.loc[(df_2['diff']>3)&(df_2.e_cal>5)]
    print('python script time = {:.0f} seconds'.format(time.time() - start))

    plt.hist2d(df_2['e_cal'], df_2['diff'], np.arange(-5,100,0.1), norm=LogNorm())
    plt.xlim(0,50)
    plt.ylim(-5,100)
    plt.xlabel('Energy (keV)', ha='right', x=1.0)
    plt.ylabel('dADC/E', ha='right', y=1.0)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
        main()
