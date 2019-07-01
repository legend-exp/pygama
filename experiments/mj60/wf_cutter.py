#!/usr/bin/env python3
import os, time, json
import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm

from pygama import DataSet
from pygama.analysis.calibration import *
from pygama.analysis.histograms import *
import pygama.utils as pgu
from matplotlib.lines import Line2D
from pygama.utils import set_plot_style
set_plot_style("clint")

def main():
    """
    mj60 analysis suite
    """
    global runDB
    with open("runDB.json") as f:
        runDB = json.load(f)

    global tier_dir
    tier_dir = runDB["tier_dir"]
    global meta_dir
    meta_dir = runDB["meta_dir"]

    # Which run number  is the being analyzed
    # run = 249
    # run = 214
    # run = 204
    run = 278

    # working on analysis for the AvsE cut in mj60
    # t1df, t2df = chunker(run)
    # cutwf, t2cut = cutter(t1df, t2df, run)
    # histograms(cutwf, t2cut, run)
    histograms(run)

# def histograms(t1df, t2df, run):
def histograms(run):
    ds = DataSet(runlist=[run], md='./runDB.json', tier_dir=tier_dir)
    t2 = ds.get_t2df()
    t2df = os.path.expandvars('{}/Spectrum_{}.hdf5'.format(meta_dir,run))
    t2df = pd.read_hdf(t2df, key="df")
    print(t2.columns)
    exit()

    # n = "tslope_savgol"
    n = "current_max"
    # n = "tslope_pz"
    # n = "tail_tau"
    # n = "tail_amp"

    e = "e_cal"
    x = t2df[e]
    # y = t2df[n]
    y = t2df[n] / x


    plt.clf()
    # H, xedges, yedges = np.histogram2d(t2df["tail_tau"], t2df["e_ftp"], bins=[2000,200], range=[[0, 6600], [0, 5]])
    plt.hist2d(x, y, bins=[1000,200], range=[[0, 2000], [0, .1]], norm=LogNorm(), cmap='jet')
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

    # plt.title("Run {}".format(run))
    # plt.xlabel("Energy (keV)", ha='right', x=1)
    # plt.ylabel(n, ha='right', y=1)
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

if __name__=="__main__":
    main()
