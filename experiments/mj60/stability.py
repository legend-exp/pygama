#!/usr/bin/env python3
import os, time, json
import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from scipy.integrate import quad
import tinydb as db
import argparse
import scipy.signal as signal

from pygama import DataSet
from pygama.analysis.calibration import *
from pygama.analysis.histograms import *
import pygama.utils as pgu
from matplotlib.lines import Line2D
from pygama.utils import set_plot_style
set_plot_style("clint")

def main():


    run_db, cal_db = "runDB.json", "calDB.json"

    par = argparse.ArgumentParser(description="calibration suite for MJ60")
    arg, st, sf = par.add_argument, "store_true", "store_false"
    arg("-ds", nargs='*', action="store", help="load runs for a DS")
    arg("-r", "--run", nargs=1, help="load a single run")

    args = vars(par.parse_args())

    # -- declare the DataSet --
    if args["ds"]:
        ds_lo = int(args["ds"][0])
        try:
            ds_hi = int(args["ds"][1])
        except:
            ds_hi = None
        ds = DataSet(ds_lo, ds_hi,
                     md=run_db, cal=cal_db)

    if args["run"]:
        ds = DataSet(run=int(args["run"][0]), sub='none',
                     md=run_db, cal=cal_db)

    # gain_shift(ds)
    # get_power_spectrum(ds)

    ########PLOT Kr vs Kr, and BKG vs BKG against each campain
    plot_psd()

def gain_shift(ds):

    calDB = ds.calDB
    query = db.Query()
    table = calDB.table("cal_pass1")
    vals = table.all()
    df_cal = pd.DataFrame(vals) # <<---- omg awesome
    data_set = ds.ds_list
    df_cal = df_cal.loc[df_cal.ds.isin(ds.ds_list)]
    data_set = df_cal.to_numpy()[:,0]
    constants = df_cal.to_numpy()[:,1]
    std = df_cal.to_numpy()[:,2]

    ylow = np.mean(constants) - .05 * np.mean(constants)
    yup = np.mean(constants) + .05 * np.mean(constants)

    plt.errorbar(data_set, constants, yerr=std, fmt='o', markersize='4', elinewidth=1)#, ecolor='b')
    plt.xlabel('Dataset')
    plt.ylabel('Calibration Constant')
    plt.ylim(ylow,yup)
    plt.title('Campaign 1')
    plt.tight_layout()
    plt.show()


def plot_psd():

    run280 = np.load('psd_280.npz')
    run296 = np.load('psd_296.npz')
    run316 = np.load('psd_316.npz')
    run330 = np.load('psd_330.npz')
    run354 = np.load('psd_354.npz')
    freq = run280['arr_0']
    psd280 = run280['arr_1']
    psd296 = run296['arr_1']
    psd316 = run316['arr_1']
    psd330 = run330['arr_1']
    psd354 = run354['arr_1']



    run946 = np.load('psd_946.npz')
    run994 = np.load('psd_994.npz')
    run1042 = np.load('psd_1042.npz')
    run1091 = np.load('psd_1091.npz')
    run1140 = np.load('psd_1140.npz')
    # freq = run946['arr_0']
    psd946 = run946['arr_1']
    psd994 = run994['arr_1']
    psd1042 = run1042['arr_1']
    psd1091 = run1091['arr_1']
    psd1140 = run1140['arr_1']

    psd_c1 = psd280 + psd296 + psd316
    psd_c2 = psd994 + psd1091

    plt.semilogy(freq, psd_c1, linewidth=2, label='C1')
    plt.semilogy(freq, psd_c2, linewidth=2, label='C2')

    # plt.semilogy(freq, psd330, linewidth=2, label='BKG_1')
    # plt.semilogy(freq, psd946, linewidth=2, label='BKG_2')

    # plt.semilogy(freq, psd280, linewidth=2, label='run280')
    # plt.semilogy(freq, psd946, linewidth=2, label='run946')
    # plt.semilogy(freq, psd1042, linewidth=2, label='run1042')
    # plt.semilogy(freq, psd1091, linewidth=2, label='run1091')
    # plt.semilogy(freq, psd1140, linewidth=2, label='run1140')
    plt.xlabel('Frequency (Hz)', ha='right', x=0.9)
    plt.ylabel('PSD (ADC^2 / Hz)', ha='right', y=1)
    plt.title(' BKG Campaign 1 vs Campaign 2')
    plt.legend(loc=1)
    plt.tight_layout()
    plt.show()

def get_power_spectrum(ds):

    t1 = ds.get_t1df()
    t1.reset_index(inplace=True)
    # key = "/ORSIS3302DecoderForEnergy"
    # wf_chunk = pd.read_hdf(t1, key, where="ievt < {}".format(75000))
    # key = "/ORSIS3302DecoderForEnergy"

    icols = []
    for idx, col in enumerate(t1.columns):
        if isinstance(col, int):
            icols.append(col)
    wfs = t1[icols].values
    wfs = wfs[:20000]

    # xvals = np.arange(0,3000)
    # start = time.time()
    # for i in range(0,5):
    # # for i in range(0,5):
    #     plt.plot(xvals, wfs[i], lw=1)
    #     plt.xlabel('Sample Number', ha='right', x=1.0)
    #     plt.ylabel('ADC Value', ha='right', y=1.0)
    #     plt.tight_layout()
    #     plt.show()

# def psd(waves, calcs, ilo=None, ihi=None, nseg=100, test=False):
    """
    calculate the psd of a bunch of wfs, and output them as a block,
    so some analysis can add them all together.
    nperseg = 1000 has more detail, but is slower
    """
    # wfs = waves["wf_blsub"]
    # if ilo is not None and ihi is not None:
    #     wfs = wfs[:, ilo:ihi]
    clk = 100e6 # Hz
    print("check2")
    nseg = 1000
    f, p = signal.welch(wfs, clk, nperseg=nseg)
    print("check3")


        # plt.semilogy(f, p[3], '-k', alpha=0.4, label='one wf')

    ptot = np.sum(p, axis=0)
    y = ptot / wfs.shape[0]
    plt.semilogy(f, y, '-b', label='all wfs')

    plt.xlabel('Frequency (Hz)', ha='right', x=0.9)
    plt.ylabel('PSD (ADC^2 / Hz)', ha='right', y=1)
    plt.legend(loc=1)
    plt.tight_layout()
    plt.show()
    np.savez("./psd_{}.npz".format(ds.runs[0]), f, y)
    exit()

    return {"psd": p, "f_psd": f}


def blsub(waves, calcs, blest="", wfin="waveform", wfout="wf_blsub", test=False):
    """
    return an ndarray of baseline-subtracted waveforms,
    using the results from the fit_bl calculator
    """
    wfs = waves[wfin]
    nwfs, nsamp = wfs.shape[0], wfs.shape[1]

    if blest == "fcdaq":
        bl_0 = calcs["fcdaq"].values[:, np.newaxis]
        blsub_wfs = wfs - bl_0

    else:
        bl_0 = calcs["bl_p0"].values[:, np.newaxis]
        if "bl_p1" in calcs.keys():
            slope_vals = calcs["bl_p1"].values[:, np.newaxis]
            bl_1 = np.tile(np.arange(nsamp), (nwfs, 1)) * slope_vals
            blsub_wfs = wfs - (bl_0 + bl_1)
        else:
            blsub_wfs = wfs - bl_0

    if test:
      iwf = 2
      while True:
        if iwf != 2:
          inp = input()
          if inp == "q": exit()
          if inp == "p": iwf -= 2
          if inp.isdigit(): iwf = int(inp) - 1
        iwf += 1
        print(iwf)
        plt.cla()

        plt.plot(np.arange(nsamp), wfs[iwf], '-g', label="raw")
        plt.plot(np.arange(nsamp), blsub_wfs[iwf], '-b', label="bl_sub")
        # plt.plot(np.arange(nsamp), blsub_avgs[iwf], '-g', label="bl_avg")

        plt.xlabel("clock ticks", ha='right', x=1)
        plt.ylabel("ADC", ha='right', y=1)
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)
        plt.grid(True)
        plt.pause(0.01)

    # note, floats are gonna take up more memory
    return {wfout: blsub_wfs}



if __name__=="__main__":
    main()
