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

    # Which run number is the being analyzed
    # run = 249
    # run = 214
    # run = 204
    run = 278
    dataset = 31

    histograms(run, dataset)

def histograms(run, dataset):

    # ds = DataSet(runlist=[191], md='./runDB.json', tier_dir=tier_dir)
    ds = DataSet(ds_lo=18, md='./runDB.json', tier_dir=tier_dir)
    t2 = ds.get_t2df()
    # t2df = os.path.expandvars('{}/Spectrum_{}.hdf5'.format(meta_dir,run))
    # t2df = pd.read_hdf(t2df, key="df")
    # t2df = t2df.reset_index(drop=True)
    t2 = t2.reset_index(drop=True)
    # a = 0.40732860520094566
    b = -1.1128841607564937
    a = 4.04702e-01
    # x = a * np.asarray(t2["e_ftp"]) + b
    cal = a * np.asarray(t2["e_ftp"])


    n = "current_max"
    e = "e_cal"
    e_over_unc = cal / np.asarray(t2["e_ftp"])
    # x = e_cal
    y0 = np.asarray(t2[n])
    y = y0 * e_over_unc / cal

    # hist, bins = np.histogram(x, bins=500, range=[1500,1700])
    hist, bins = np.histogram(cal, bins=450, range=[1530,1620])
    hist = hist * 5

    def gauss(x, *params):
        y = np.zeros_like(x)
        for i in range(0, len(params) - 1, 3):
            x0 = params[i]
            a = params[i + 1]
            sigma = params[i + 2]
            y += a * np.exp(-(x - x0)**2 / (2 * sigma**2))
        y = y + params[-1]
        return y

    p0_list = [1590, 200, 3, 4]
    # bnd = ((p0_list[0]*.95, 1, 0, 0),
    #         (p0_list[0]*1.05, 500, 100, 500))

    par, pcov = curve_fit(
        gauss, bins[1:], hist, p0=p0_list) #bounds=bnd)
    print(par)
    perr = np.sqrt(np.diag(pcov))
    print(perr)



    mu, amp, sig, bkg = par[0], par[1], par[2], par[-1]
    print("Scanning peak ", n, " at energy", mu)
    ans = quad(gauss, 1583, 1600, args=(mu, amp, sig, bkg))
    answer = ans[0] - ((1600-1583)*bkg)
    print("Counts in ", mu, " peak is ", answer)

    cut = answer
    line = .012

    y1 = y[np.where(line < y)]
    x1 = cal[np.where(line < y)]
    # hist1, bins1 = np.histogram(x1, bins=500, range=[1500,1700])
    hist1, bins1 = np.histogram(x1, bins=450, range=[1530,1620])
    hist1 = hist1*5

    while cut > .9 * answer:

        y1 = y[np.where(line < y)]
        x1 = cal[np.where(line < y)]

        # hist1, bins1 = np.histogram(x1, bins=500, range=[1500,1700])
        hist1, bins1 = np.histogram(x1, bins=450, range=[1530,1620])
        hist1 = hist1*5

        par1, pcov1 = curve_fit(
            gauss, bins1[1:], hist1, p0=p0_list)
        # print(par1)
        perr1 = np.sqrt(np.diag(pcov))
        # print(perr1)

        mu1, amp1, sig1, bkg1 = par1[0], par1[1], par1[2], par1[-1]
        # print("Scanning cut peak ", n, " at energy", mu1)
        ans1 = quad(gauss, 1583, 1600, args=(mu1, amp1, sig1, bkg1))
        cut = ans1[0] - ((1600-1583)*bkg1)
        # print("Counts in ", mu1, " peak is ", cut)

        line += .00001

    print(line, cut)
    plt.hist2d(cal, y, bins=[1000,200], range=[[0, 2000], [0, .05]], norm=LogNorm(), cmap='jet')
    plt.hlines(line, 0, 2000, color='r', linewidth=1.5)
    cbar = plt.colorbar()
    plt.title("Dataset {}".format(dataset))
    plt.xlabel("Energy (keV)", ha='right', x=1)
    plt.ylabel("A/Eunc", ha='right', y=1)
    cbar.ax.set_ylabel('Counts')
    plt.tight_layout()
    plt.show()

    hist, bins = np.histogram(cal, bins=2000, range=[0,2000])
    hist1, bins1 = np.histogram(x1, bins=2000, range=[0,2000])
    plt.clf()
    plt.plot(bins[1:], hist, color='black', ls="steps", linewidth=1.5, label='Calibrated Energy: Dataset {}'.format(dataset))
    plt.plot(bins1[1:], hist1, '-r', ls="steps", linewidth=1.5, label='AvsE Cut: Dataset {}'.format(dataset))
    # plt.plot(bins[1:], gauss(bins[1:], *par), color='blue')
    # plt.plot(bins1[1:], gauss(bins1[1:], *par1), color='aqua')
    plt.ylabel('Counts')
    plt.xlabel('keV')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    main()
