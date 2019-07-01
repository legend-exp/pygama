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


    # histograms(cutwf, t2cut, run)
    histograms(run)

# def histograms(t1df, t2df, run):
def histograms(run):
    ds = DataSet(runlist=[run], md='./runDB.json', tier_dir=tier_dir)
    t2 = ds.get_t2df()
    t2df = os.path.expandvars('{}/Spectrum_{}.hdf5'.format(meta_dir,run))
    t2df = pd.read_hdf(t2df, key="df")
    t2df = t2df.reset_index(drop=True)
    t2 = t2.reset_index(drop=True)
    # print(t2.columns)
    # exit()

    n = "current_max"
    e = "e_cal"
    e_over_unc = np.asarray(t2df[e]) / np.asarray(t2["e_ftp"])
    x = np.asarray(t2df[e])
    y0 = np.asarray(t2df[n])
    y = y0 * e_over_unc / x



    hist, bins = np.histogram(x, bins=500, range=[1500,1700])

    def gauss(x, *params):
        y = np.zeros_like(x)
        for i in range(0, len(params) - 1, 3):
            x0 = params[i]
            a = params[i + 1]
            sigma = params[i + 2]
            y += a * np.exp(-(x - x0)**2 / (2 * sigma**2))
        y = y + params[-1]
        return y

    p0_list= [1590, 31, 3, 4]

    par, pcov = curve_fit(
        gauss, bins[1:], hist, p0=p0_list)
    print(par)
    perr = np.sqrt(np.diag(pcov))
    print(perr)

    mu, amp, sig, bkg = par[0], par[1], par[2], par[-1]
    print("Scanning peak ", n, " at energy", mu)
    ans = quad(gauss, mu - 5 * sig, mu + 5 * sig, args=(mu, amp, sig, bkg))
    answer = ans[0]
    print("Counts in ", mu, " peak is ", answer)

    plt.plot(bins[1:], hist, ls="steps", linewidth=2, label='Calibrated Energy: Run {}'.format(run))
    plt.plot(bins[1:], gauss(bins[1:], *par), '-r')
    plt.legend()
    plt.tight_layout()
    plt.show()

    y1 = y[np.where(.021 < y)]
    x1 = x[np.where(.021 < y)]





    plt.clf()
    plt.hist2d(x1, y1, bins=[1000,200], range=[[0, 2000], [0, .1]], norm=LogNorm(), cmap='jet')
    cbar = plt.colorbar()
    plt.title("Run {}".format(run))
    plt.xlabel("Energy (keV)", ha='right', x=1)
    plt.ylabel("A*E/Eunc", ha='right', y=1)
    cbar.ax.set_ylabel('Counts')
    plt.tight_layout()
    plt.show()






if __name__=="__main__":
    main()
