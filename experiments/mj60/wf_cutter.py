#!/usr/bin/env python3
import os, time, json
import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
import scipy.signal as signal
import argparse

from pygama import DataSet
from pygama.analysis.calibration import *
from pygama.analysis.histograms import *
import pygama.utils as pgu
from matplotlib.lines import Line2D
from pygama.utils import set_plot_style
set_plot_style("clint")

def main():
    """
    mj60 waveform viewer
    """
    run_db, cal_db = "runDB.json", "calDB.json"

    par = argparse.ArgumentParser(description="waveform viewer for mj60")
    arg, st, sf = par.add_argument, "store_true", "store_false"
    arg("-ds", nargs='*', action="store", help="load runs for a DS")
    arg("-r", "--run", nargs=1, help="load a single run")
    arg("-db", "--writeDB", action=st, help="store results in DB")
    args = vars(par.parse_args())

    # -- declare the DataSet --
    if args["ds"]:
        ds_lo = int(args["ds"][0])
        try:
            ds_hi = int(args["ds"][1])
        except:
            ds_hi = None
        ds = DataSet(ds_lo, ds_hi,
                     md=run_db, cal = cal_db) #,tier_dir=tier_dir)

    if args["run"]:
        ds = DataSet(run=int(args["run"][0]),
                     md=run_db, cal=cal_db)


    # Which run number is the being analyzed
    # run = 249
    # run = 214
    # run = 204
    # run = 278

    # working on analysis for the AvsE cut in mj60
    # t1df, t2df = chunker(run)
    # cutwf, t2cut = cutter(t1df, t2df, run)
    # histograms(cutwf, t2cut, run)
    # histograms(ds)
    drift_correction(ds)

# def histograms(t1df, t2df, run):
def histograms(ds):

    t2 = ds.get_t2df()
    print(t2.columns)
    exit()
    t2df = os.path.expandvars('{}/Spectrum_{}.hdf5'.format(meta_dir,run))
    t2df = pd.read_hdf(t2df, key="df")


    # n = "tslope_savgol"
    # n = "current_max"
    # n = "tslope_pz"
    n = "tail_tau"
    # n = "tail_amp"

    e = "e_cal"
    x = t2df[e]
    # y = t2df[n]
    y = t2df[n] / x


    plt.clf()
    # H, xedges, yedges = np.histogram2d(t2df["tail_tau"], t2df["e_ftp"], bins=[2000,200], range=[[0, 6600], [0, 5]])
    plt.hist2d(x, y, bins=[1000,200], range=[[0, 200], [0, .001]], norm=LogNorm(), cmap='jet')
    # plt.hist2d(x, y, bins=[1000,1000], norm=LogNorm())
    # plt.scatter(H[0],H[1])

    # f = plt.figure(figsize=(20,5))
    # p1 = f.add_subplot(111, title='Test', xlabel='Energy (keV)', ylabel=n)
    # h1,xedg1,yedg1 = np.histogram2d(x, y, bins=[1000,200], range=[[0,2000],[0,100]])
    # h1 = h1.T
    # # hMin, hMax = np.amin(h1), np.amax(h1)
    # # im1 = p1.imshow(h1,cmap='jet',vmin=hMin,vmax=hMax, aspect='auto') #norm=LogNorm())
    # im1 = p1.imshow(h1,cmap='jet', origin='lower', aspect='auto', norm=LogNorm(), extent=[xedg1[0], xedg1[-1], yedg1[0], yedg1[-1]])

    # cb1 = f.colorbar(im1, ax=p1)#, fraction=0.037, pad=0.04)

    cbar = plt.colorbar()

    # plt.xscale('symlog')
    # plt.yscale('symlog')

    plt.title("Run {}".format(run))
    plt.xlabel("Energy (keV)", ha='right', x=1)
    plt.ylabel(n, ha='right', y=1)
    # cbar.ax.set_ylabel('Counts')
    # plt.ylabel("tslope_savgol", ha='right', y=1)
    # plt.ylabel("A/E_ftp", ha='right', y=1)
    # plt.tight_layout()
    # # plt.savefig('./plots/meeting_plots/run{}_{}_vs_{}.png'.format(run, n, e))
    # plt.show()

    # xlo, xhi, xpb = 0, 10000, 10
    # xP, hP = get_hist(t2df["trap_max"], xlo, xhi, xpb)
    #
    # plt.plot(xP, hP, ls='steps', lw=1.5, c='m',
    #          label="pygama trap_max, {} cts".format(sum(hP)))
    # plt.xlabel("Energy (uncal)", ha='right', x=1)
    # plt.ylabel("Counts", ha='right', y=1)
    # plt.legend()
    plt.tight_layout()
    plt.show()

def chunker(run):

    t1df = os.path.expandvars('{}/t1_run{}.h5'.format(tier_dir,run))
    t2df = os.path.expandvars('{}/Spectrum_{}.hdf5'.format(meta_dir,run))
    t2df = pd.read_hdf(t2df, key="df")
    t2df_chunk = t2df[:75000]
    key = "/ORSIS3302DecoderForEnergy"
    wf_chunk = pd.read_hdf(t1df, key, where="ievt < {}".format(75000))
    wf_chunk.reset_index(inplace=True) # required step -- fix pygama "append" bug
    t2df = t2df.reset_index(drop=True)

    # create waveform block.  mask wfs of unequal lengths
    icols = []
    for idx, col in enumerate(wf_chunk.columns):
        if isinstance(col, int):
            icols.append(col)
    wf_block = wf_chunk[icols].values
    # print(wf_block.shape, type(wf_block))
    # print(t2df_chunk)

    return wf_block, t2df_chunk

def cutter(t1df, t2df, run):

    # t2cut = t2df.loc[(t2df.e_cal>3.1099]
    t2cut = t2df
    print(t2cut.index)
    print(t2cut)
    cutwf = t1df[t2cut.index]
    print(cutwf)


    # xvals = np.arange(0,3000)
    # start = time.time()
    # for i in range(len(t2cut.index)):
    # # for i in range(0,5):
    #     plt.plot(xvals, cutwf[i], lw=1)
    #     plt.xlabel('Sample Number', ha='right', x=1.0)
    #     plt.ylabel('ADC Value', ha='right', y=1.0)
    #     plt.tight_layout()
    #     plt.show()

    return cutwf, t2cut

def drift_correction(ds):

    ## testing a drift time correction code

    t1df = ds.get_t1df()
    t1df.reset_index(inplace=True)
    t2df = ds.get_t2df()

    # key = "/ORSIS3302DecoderForEnergy"
    # wf_chunk = pd.read_hdf(t1df, key, where="ievt < {}".format(75000))
    # wf_chunk.reset_index(inplace=True) # required step -- fix pygama "append" bug
    # t2df = t2df.reset_index(drop=True)

    # create waveform block.  mask wfs of unequal lengths

    number = 20000
    icols = []
    for idx, col in enumerate(t1df.columns):
        if isinstance(col, int):
            icols.append(col)
    wfs = t1df[icols].values
    wfs = wfs[:number]
    t2df_chunk = t2df[:number]
    # print(wf_block.shape, type(wf_block))
    # print(t2df_chunk)
    t0 = t2df_chunk['t0']
    energy = t2df_chunk['e_ftp']

    baseline = wfs[:, 0:500]
    avg_bl = []
    for i in range(0,number):
        avg_bl.append(np.mean(baseline[i], keepdims=True))
    avg_bl = np.asarray(avg_bl)
    wfs = np.asarray(wfs)
    wfs = wfs - avg_bl

    clk = 100e6
    decay = 82
    wfs = pz(wfs, decay, clk)

    xvals = np.arange(0,3000)
    start = time.time()
    for i in range(0,5):
    # for i in range(0,5):
        plt.plot(xvals, wfs[i], lw=1)
        plt.vlines(t0[i], np.amin(wfs[i]), np.amax(wfs[i]), color='r', linewidth=1.5)
        plt.hlines(energy[i], 0, 3000, color='r', linewidth=1.5, zorder=10)
        plt.xlabel('Sample Number', ha='right', x=1.0)
        plt.ylabel('ADC Value', ha='right', y=1.0)
        plt.tight_layout()
        plt.show()

def pz(wfs, decay, clk):
    """
    pole-zero correct a waveform
    decay is in us, clk is in Hz
    """


    # get linear filter parameters, in units of [clock ticks]
    dt = decay * (1e10 / clk)
    rc = 1 / np.exp(1 / dt)
    num, den = [1, -1], [1, -rc]

    # reversing num and den does the inverse transform (ie, PZ corrects)
    pz_wfs = signal.lfilter(den, num, wfs)


    return pz_wfs

    # return wfs, t2df_chunk

if __name__=="__main__":
    main()
