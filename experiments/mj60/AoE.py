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

#Code to find and record the optimal A/E cut
def find_cut(ds, ds_lo, write_db=False):

    #Make tier2 dataframe, get e_ftp and first pass calibration constants, then calibrate
    t2 = ds.get_t2df()
    t2 = t2.reset_index(drop=True)

    #Get pass1 calibration constant TODO: need pass2 constants at some point
    calDB = ds.calDB
    query = db.Query()
    table = calDB.table("cal_pass1")
    vals = table.all()
    df_cal = pd.DataFrame(vals) # <<---- omg awesome
    df_cal = df_cal.loc[df_cal.ds==ds_lo]
    p1cal = df_cal.iloc[0]["p1cal"]
    cal = p1cal * np.asarray(t2["e_ftp"])

    #Make A/E array
    current = "current_max"
    e_over_unc = cal / np.asarray(t2["e_ftp"]) #Needed to normalize or something, idk
    y0 = np.asarray(t2[current])
    a_over_e = y0 * e_over_unc / cal

    y = linear_correction(cal, a_over_e) # Linear correct slight downward trend

    # Two separate functions, one for Ac contaminated peak(Th232), one for Th228
    ans = input('Are you running A/E on Th232? \n y/n -->')
    if ans == 'y':
        th_232(cal, y, ds)
    else:
        regular_cut(cal, y, ds)

    # Write cut to the calDB.json file
    if write_db:
        table = calDB.table("A/E_cut")
        for dset in ds.ds_list:
            row = {"ds":dset, "line":line}
            table.upsert(row, query.ds == dset)


def linear_correction(energy, a_over_e):

##################
### TODO: Use compt continuum bkg areas with a gaussian fit to
##################

    max_list = []
    peak_list = np.asarray([2614.5, 1460.8, 583.2])
    for peak in peak_list:

        aoe = a_over_e[np.where((energy > (peak-20)) & (energy < (peak + 20)))]
        hist, bins = np.histogram(aoe, bins=200, range=[0.01,0.03])
        b = (bins[:-1] + bins[1:]) / 2

        max_c = b[0]
        max = hist[0]
        for i in range(len(b)):   # Find max point of A/E dist
            if max < hist[i]:
                max = hist[i]
                max_c = b[i]
        max_list.append(max_c)

    max_list = np.asarray(max_list)

    def line(x, a, b):
        return a * x + b
    par, pcov = curve_fit(line, peak_list, max_list)

    print(par)

    a_over_e = a_over_e / (par[0] * energy + par[1])

    # for i in range(len(a_over)):
    #     a_over[i] = a_over[i] / (par[0] * energy[i] + par[1])

    return a_over_e



def regular_cut(cal, y, ds):

## Find A/E cut for Th228 source

    dep_range = [1530,1620]
    hist, bins = np.histogram(cal, bins=450, range=dep_range)
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

    p0_list = [1591, 200, 3, 4]

    par, pcov = curve_fit(
        gauss, bins[1:], hist, p0=p0_list)
    print(par)
    perr = np.sqrt(np.diag(pcov))
    print(perr)

    mu, amp, sig, bkg = par[0], par[1], par[2], par[-1]
    print("Scanning ", mu, " peak")
    ans = quad(gauss, 1583, 1600, args=(mu, amp, sig, bkg))
    counts = ans[0] - ((1600-1583)*bkg)
    print("Counts in ", mu, " peak is ", counts)

    cut = counts
    line = .4

    y1 = y[np.where(line < y)]
    x1 = cal[np.where(line < y)]
    # hist1, bins1 = np.histogram(x1, bins=500, range=[1500,1700])
    hist1, bins1 = np.histogram(x1, bins=450, range=[1530,1620])
    hist1 = hist1*5

    print("Finding optimal cut, keeping 90% of 1592 DEP")
    while cut > .9 * counts:

        y1 = y[np.where(line < y)]
        x1 = cal[np.where(line < y)]

        hist1, bins1 = np.histogram(x1, bins=450, range=dep_range)
        hist1 = hist1*5

        par1, pcov1 = curve_fit(
            gauss, bins1[1:], hist1, p0=p0_list)
        perr1 = np.sqrt(np.diag(pcov1))

        mu1, amp1, sig1, bkg1 = par1[0], par1[1], par1[2], par1[-1]
        ans1 = quad(gauss, 1583, 1600, args=(mu1, amp1, sig1, bkg1))
        cut = ans1[0] - ((1600-1583)*bkg1)

        line += .001


    print(line, cut)
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

    plt.clf()
    plt.semilogy(bins[1:], hist, color='black', ls="steps", linewidth=1.5, label='Calibrated Energy: Dataset {}'.format(ds_lo))
    plt.semilogy(bins1[1:], hist1, '-r', ls="steps", linewidth=1.5, label='AvsE Cut: Dataset {}'.format(ds_lo))
    plt.ylabel('Counts')
    plt.xlabel('keV')
    plt.legend()
    plt.tight_layout()
    plt.show()



def th_232(energy, a_over_e, ds, write_db=False):

    ## Find A/E cut for Th232 source

    # file1 = np.load('./ds18.npz')
    # file2 = np.load('./bins_ds18.npz')
    # counts = file1['arr_0']
    # energy = file2['arr_0']

    #FWHM of nearby peaks is 2.5
    dep_range = [1530,1620]
    hist, bins = np.histogram(energy, bins=(dep_range[1]-dep_range[0]), range=dep_range)
    b = (bins[:-1] + bins[1:]) / 2

    def gauss(x, *params):
        y = np.zeros_like(x)
        for i in range(0, len(params) - 1, 3):
            x0 = params[i]
            a = params[i + 1]
            sigma = params[i + 2]
            y += a * np.exp(-(x - x0)**2 / (2 * sigma**2))
        y = y + params[-1]
        return y

    p0_list = [1588.2, 400, 2.5, 1592.5, 400, 2.5, 157]
    bnds = ([1588, 0, .9*p0_list[2], 1592.5, 0, .9*p0_list[5], 0],
            [1589, 700, 1.1*p0_list[2], 1593, 700, 1.1*p0_list[5], 300])

    par, pcov = curve_fit(gauss, b, hist, p0=p0_list, bounds=bnds)
    print(par)
    perr = np.sqrt(np.diag(pcov))
    print(perr)

    # np.savez('double_gauss_params', par)

    plt.title('Peak 1590 combined')
    plt.plot(b, hist, ls="steps", color='black')
    plt.plot(b, gauss(b, *par), '-r')
    plt.tight_layout()
    plt.show()

    ac_peak_height = par[1]
    th_peak_height = par[4]
    cut_ac_peak_height = par[1]
    cut_th_peak_height = par[4]
    ss_eff_array = []
    ms_eff_array = []
    cut_line_list = []

    line = .4
    print("Finding optimal cut, keeping 90% of 1592 DEP")

    while cut_th_peak_height > .9 * th_peak_height:

        y = a_over_e[np.where(line < a_over_e)]
        e1 = energy[np.where(line < a_over_e)]

        hist1, bins1 = np.histogram(e1, bins=(dep_range[1]-dep_range[0]), range=dep_range)

        par1, pcov1 = curve_fit(
            gauss, b, hist1, p0=p0_list, bounds=bnds)
        perr1 = np.sqrt(np.diag(pcov1))

        cut_ac_peak_height = par1[1]
        cut_th_peak_height = par1[4]
        ss_eff = cut_th_peak_height / th_peak_height
        ms_eff = cut_ac_peak_height / ac_peak_height
        ss_eff_array.append(ss_eff)
        ms_eff_array.append(ms_eff)
        cut_line_list.append(line)

        line += .001

    print(line)

    plt.clf()
    plt.hist2d(energy, a_over_e, bins=[1000,200], range=[[0, 2000], [0, 2]], norm=LogNorm(), cmap='jet')
    plt.hlines(line, 0, 2000, color='r', linewidth=1.5)
    plt.xlabel("Energy (keV)", ha='right', x=1)
    plt.ylabel("A/Eunc", ha='right', y=1)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts')
    plt.tight_layout()
    plt.show()

    plt.clf()
    a1 = a_over_e[np.where((1589 < energy) & (energy < 1595))]
    hist, bins = np.histogram(a_over_e, bins = 200, range=[.4,1.5])
    plt.vlines(line, 0, 300000, color='r', linewidth=1.5)
    plt.plot(bins[1:], hist)
    plt.xlabel('A over E normalized')
    plt.ylabel('Counts')
    plt.show()

    plt.clf()
    plt.plot(cut_line_list, ss_eff_array, label='ss_eff')
    plt.plot(cut_line_list, ms_eff_array, label='ms_eff')
    plt.ylabel('eff')
    plt.xlabel('AoverE_normalized')
    plt.legend()
    plt.show()

    hist, bins = np.histogram(energy, bins=2600, range=[0,2600])
    hist1, bins1 = np.histogram(e1, bins=2600, range=[0,2600])

    plt.clf()
    plt.semilogy(bins[1:], hist, color='black', ls="steps", linewidth=1.5, label='Calibrated Energy: Dataset {}'.format(ds.ds_list[0]))
    plt.semilogy(bins1[1:], hist1, '-r', ls="steps", linewidth=1.5, label='AvsE Cut: Dataset {}'.format(ds.ds_list[0]))
    plt.ylabel('Counts')
    plt.xlabel('keV')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return line

    #b + a1e^-(alphaE-1588.2 +delta)+ a2e^-(alphaE-1592.5 +delta) fit this with
    #same delta or alpha for each one



    #fit whole 1590 peak region with two gaussians

    #start cutting up AoverE line, and fitting region on ms and ss, find the efficiency
    # of both ms and ss, find where there is 90% of the counts left from the original
    #1592 gaussian from above


if __name__=="__main__":
    main()
