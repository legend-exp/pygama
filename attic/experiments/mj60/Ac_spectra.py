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
    Code to implement an A/E cut
    """
    # global runDB
    # with open("runDB.json") as f:
    #     runDB = json.load(f)

    # global tier_dir
    # tier_dir = runDB["tier_dir"]
    # global meta_dir
    # meta_dir = runDB["meta_dir"]

    run_db, cal_db = "runDB.json", "calDB.json"

    par = argparse.ArgumentParser(description="A/E cut for MJ60")
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

    find_cut(ds, ds_lo, args["writeDB"])
    # ac_spectra()

#Code to find and record the optimal A/E cut
def find_cut(ds, ds_lo, write_db=False):

    #Make tier2 dataframe, get e_ftp and first pass calibration constants, then calibrate
    t2 = ds.get_t2df()
    t2 = t2.reset_index(drop=True)

    calDB = ds.calDB
    query = db.Query()
    table = calDB.table("cal_pass1")
    vals = table.all()
    df_cal = pd.DataFrame(vals) # <<---- omg awesome
    df_cal = df_cal.loc[df_cal.ds==ds_lo]
    p1cal = df_cal.iloc[0]["p1cal"]
    cal = p1cal * np.asarray(t2["e_ftp"])

    hist, bins = np.histogram(cal, bins=2000, range=[0,2000])
    b = (bins[:-1] + bins[1:]) / 2
    # np.savez('ds{}'.format(ds_lo), cal)
    # np.savez('bins_ds{}'.format(ds_lo), b)


    # plt.clf()
    plt.title('DS{}'.format(ds_lo))
    plt.plot(b, hist, ls="steps", linewidth=1.5)
    plt.ylabel('Counts')
    plt.xlabel('keV')
    plt.tight_layout()
    plt.show()
    exit()


    current = "current_max"
    e_over_unc = cal / np.asarray(t2["e_ftp"])
    y0 = np.asarray(t2[current])
    a_over_e = y0 * e_over_unc / cal

    y = linear_correction(cal, a_over_e)

    # dep_range = [1530,1620]
    # hist, bins = np.histogram(cal, bins=450, range=dep_range)
    # hist = hist * 5
    #
    def gauss(x, *params):
        y = np.zeros_like(x)
        for i in range(0, len(params) - 1, 3):
            x0 = params[i]
            a = params[i + 1]
            sigma = params[i + 2]
            y += a * np.exp(-(x - x0)**2 / (2 * sigma**2))
        y = y + params[-1]
        return y
    #
    # p0_list = [1591, 200, 3, 4]
    #
    # par, pcov = curve_fit(
    #     gauss, bins[1:], hist, p0=p0_list)
    # print(par)
    # perr = np.sqrt(np.diag(pcov))
    # print(perr)
    #
    # mu, amp, sig, bkg = par[0], par[1], par[2], par[-1]
    # print("Scanning ", mu, " peak")
    # ans = quad(gauss, 1583, 1600, args=(mu, amp, sig, bkg))
    # counts = ans[0] - ((1600-1583)*bkg)
    # print("Counts in ", mu, " peak is ", counts)
    #
    # cut = counts
    # line = .4
    #
    # y1 = y[np.where(line < y)]
    # x1 = cal[np.where(line < y)]
    # # hist1, bins1 = np.histogram(x1, bins=500, range=[1500,1700])
    # hist1, bins1 = np.histogram(x1, bins=450, range=[1530,1620])
    # hist1 = hist1*5
    #
    # print("Finding optimal cut, keeping 90% of 1592 DEP")
    # while cut > .9 * counts:
    #
    #     y1 = y[np.where(line < y)]
    #     x1 = cal[np.where(line < y)]
    #
    #     hist1, bins1 = np.histogram(x1, bins=450, range=dep_range)
    #     hist1 = hist1*5
    #
    #     par1, pcov1 = curve_fit(
    #         gauss, bins1[1:], hist1, p0=p0_list)
    #     perr1 = np.sqrt(np.diag(pcov1))
    #
    #     mu1, amp1, sig1, bkg1 = par1[0], par1[1], par1[2], par1[-1]
    #     ans1 = quad(gauss, 1583, 1600, args=(mu1, amp1, sig1, bkg1))
    #     cut = ans1[0] - ((1600-1583)*bkg1)
    #
    #     line += .0005

    line = .95

    y1 = y[np.where(line < y)]
    x1 = cal[np.where(line < y)]
    y2 = y[np.where(line > y)]
    x2 = cal[np.where(line > y)]
    np.savez('thorium', x1)
    np.savez('Ac', x2)

    # print(line, cut)
    plt.hist2d(cal, y, bins=[1000,200], range=[[0, 2000], [0, 2]], norm=LogNorm(), cmap='jet')
    plt.hlines(line, 0, 2000, color='r', linewidth=1.5)
    cbar = plt.colorbar()
    plt.title("Dataset {}".format(ds_lo))
    plt.xlabel("Energy (keV)", ha='right', x=1)
    plt.ylabel("A/Eunc", ha='right', y=1)
    cbar.ax.set_ylabel('Counts')
    plt.tight_layout()
    plt.show()

    hist, bins = np.histogram(cal, bins=2000, range=[0,2000])
    hist1, bins1 = np.histogram(x1, bins=2000, range=[0,2000])

    hist2, bins2 = np.histogram(x2, bins=2000, range=[0,2000])

    p0_list = [1593, 200, 3, 4]

    par, pcov = curve_fit(
        gauss, bins1[1:], hist1, p0=p0_list)
    print(par)
    perr = np.sqrt(np.diag(pcov))
    print(perr)

    plt.clf()
    # plt.semilogy(bins[1:], hist, color='black', ls="steps", linewidth=1.5, label='Calibrated Energy: Dataset {}'.format(ds_lo))
    plt.semilogy(bins1[1:], hist1, '-r', ls="steps", linewidth=1.5, label='AvsE Cut: Dataset {}'.format(ds_lo))
    plt.ylabel('Counts')
    plt.xlabel('keV')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.clf()
    plt.semilogy(bins2[1:], hist2, ls="steps", linewidth=1.5)
    plt.title('Ac spectra')
    plt.ylabel('Counts')
    plt.xlabel('keV')
    plt.tight_layout()
    plt.show()

    if write_db:
        table = calDB.table("A/E_cut")
        for dset in ds.ds_list:
            row = {"ds":dset, "line":line}
            table.upsert(row, query.ds == dset)

def linear_correction(energy, a_over):

    max_list = []
    peak_list = np.asarray([2614.5, 1460.8, 583.2])
    for peak in peak_list:

        aoe = a_over[np.where((energy > (peak-20)) & (energy < (peak + 20)))]
        hist, bins = np.histogram(aoe, bins=200, range=[0.01,0.03])
        b = (bins[:-1] + bins[1:]) / 2

        max_c = b[0]
        max = hist[0]
        for i in range(len(b)):
            if max < hist[i]:
                max = hist[i]
                max_c = b[i]
        max_list.append(max_c)

    max_list = np.asarray(max_list)

    def line(x, a, b):
        return a * x + b
    par, pcov = curve_fit(line, peak_list, max_list)

    print(par)

    a_over = a_over / (par[0] * energy + par[1])

    # for i in range(len(a_over)):
    #     a_over[i] = a_over[i] / (par[0] * energy[i] + par[1])

    return a_over

def ac_spectra():

    file1 = np.load('./Ac.npz')
    file2 = np.load('./thorium.npz')
    x2 = file1['arr_0']
    x1 = file2['arr_0']


    def gauss(x, *params):
        y = np.zeros_like(x)
        for i in range(0, len(params) - 1, 3):
            x0 = params[i]
            a = params[i + 1]
            sigma = params[i + 2]
            y += a * np.exp(-(x - x0)**2 / (2 * sigma**2))
        y = y + params[-1]
        return y

    hist1, bins1 = np.histogram(x1, bins=2000, range=[1530,1620])
    hist2, bins2 = np.histogram(x2, bins=2000, range=[1530,1620])

    p0_list1 = [1593, 200, 3, 4]
    p0_list2 = [1589, 200, 3, 4]

    par1, pcov1 = curve_fit(
        gauss, bins1[1:], hist1, p0=p0_list1)
    print(par1)
    perr1 = np.sqrt(np.diag(pcov1))
    print(perr1)

    par2, pcov2 = curve_fit(
        gauss, bins2[1:], hist2, p0=p0_list2)
    print(par2)
    perr2 = np.sqrt(np.diag(pcov2))
    print(perr2)

    plt.clf()
    plt.title('Thorium spectra')
    plt.plot(bins1[1:], hist1, '-r', ls="steps", linewidth=1.5)
    plt.plot(bins1[1:], gauss(bins1[1:], *par1))
    plt.ylabel('Counts')
    plt.xlabel('keV')
    plt.tight_layout()
    plt.show()

    plt.clf()
    plt.plot(bins2[1:], hist2, ls="steps", linewidth=1.5)
    plt.plot(bins2[1:], gauss(bins2[1:], *par2), '-r')
    plt.title('Ac spectra')
    plt.ylabel('Counts')
    plt.xlabel('keV')
    plt.tight_layout()
    plt.show()



if __name__=="__main__":
    main()
