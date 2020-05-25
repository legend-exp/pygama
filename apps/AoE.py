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
from statsmodels.stats import proportion

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

    run_db, cal_db = "../experiments/mj60/runDB.json", "../experiments/mj60/calDB.json"

    par = argparse.ArgumentParser(description="A/E cut for MJ60")
    arg, st, sf = par.add_argument, "store_true", "store_false"
    arg("-ds", nargs='*', action="store", help="load runs for a DS")
    arg("-r", "--run", nargs=1, help="load a single run")
    arg("-db", "--writeDB", action=st, help="store results in DB")
    arg("-mode", "--mode", nargs=1, action="store",
        help="232 or 228 for the two diffrerent thorium types")
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

    find_cut(ds, ds_lo, args["mode"][0], args["writeDB"])

def find_cut(ds, ds_lo, mode, write_db=False):
    """
    Find and record (if -db is chosen) an A/E cut for either Th232 or Th228 source.
    Currently a brute force algorithm is chosen, but code is in progress for an
    optimized algorithm.
    """

    #Make tier2 dataframe
    t2 = ds.get_t2df()
    t2 = t2.reset_index(drop=True)

    #Get e_ftp and pass1 calibration constant TODO: need pass2 constants at some point
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

    #####
    # test_code(y, cal, ds)
    # exit()
    #####


    # Two separate functions, one for Ac contaminated peak(Th232), one for Th228
    if mode == '232':
        line, ss_eff, ms_eff, cc_eff, cut_list = th_232(cal, y, ds)
    elif mode == '228':
        line = regular_cut(cal, y, ds)
    else:
        print('Must specify a mode, 228 or 232')
        exit()

    # Write cut to the calDB.json file
    if write_db:
        table = calDB.table("A/E_cut")
        for dset in ds.ds_list:
            row = {"ds":dset, "line":line}
            table.upsert(row, query.ds == dset)


def regular_cut(energy, y, ds):
    """
    Algorithm for Th228 source
    """

    cc_range = [1790,2050]
    ms_range = [2083, 2123]
    dep_range = [1530,1620]
    hist, bins = np.histogram(energy, bins=(dep_range[1]-dep_range[0]),
                                range=dep_range)
    hist1, bins1 = np.histogram(energy, bins=(ms_range[1]-ms_range[0]),
                                range=ms_range)
    b = (bins[:-1] + bins[1:]) / 2
    b1 = (bins1[:-1] + bins1[1:]) / 2


    p0_list = [1592, 200, 2.5, 4]
    p1_list = [2103, 400, 3.3, 118]

    par, pcov = curve_fit(
        gauss, bins[1:], hist, p0=p0_list)
    print(par)
    perr = np.sqrt(np.diag(pcov))
    print(perr)

    par1, pcov1 = curve_fit(
        gauss, bins1[1:], hist1, p0=p1_list)
    print(par1)
    perr1 = np.sqrt(np.diag(pcov))
    print(perr1)

    #Visual check of 1592 peak
    plt.title('Peak 1592')
    plt.plot(b, hist, ds="steps", color='black')
    plt.plot(b, gauss(b, *par), '-r')
    plt.tight_layout()
    plt.show()

    plt.clf()
    plt.title('Peak 2103')
    plt.plot(b1, hist1, ds="steps", color='black')
    plt.plot(b1, gauss(b1, *par1), '-r')
    plt.tight_layout()
    plt.show()

    cc_height = np.sum(energy[np.where((energy > cc_range[0]) & (energy < cc_range[1]))]) / (cc_range[1]-cc_range[0])

    #set initial params and initial cut height
    th_peak_height = par[1]
    cut_th_peak_height = par[1]
    sep_peak_height = par1[1]
    ss_eff_array = []
    cc_eff_array = []
    ms_eff_array = []
    cut_line_list = []

    line = .6

    y1 = y[np.where(line < y)]
    x1 = energy[np.where(line < y)]
    # hist1, bins1 = np.histogram(x1, bins=500, range=[1500,1700])
    hist1, bins1 = np.histogram(x1, bins=450, range=[1530,1620])
    hist1 = hist1*5

    #Actual algorithm
    #When cut peak height over original peak height is .9, quit
    print("Finding optimal cut, keeping 90% of 1592 DEP")
    while cut_th_peak_height > .9 * th_peak_height:

        y1 = y[np.where(line < y)]
        e1 = energy[np.where(line < y)]
        cut_cc_height = np.sum(e1[np.where((e1 > cc_range[0]) & (e1 < cc_range[1]))]) / (cc_range[1]-cc_range[0])


        hist1, bins1 = np.histogram(e1, bins=(dep_range[1]-dep_range[0]),
                                    range=dep_range)
        hist2, bins2 = np.histogram(e1, bins=(ms_range[1]-ms_range[0]),
                                    range=ms_range)

        par1, pcov1 = curve_fit(
            gauss, bins1[1:], hist1, p0=p0_list)
        perr1 = np.sqrt(np.diag(pcov1))

        par2, pcov2 = curve_fit(
            gauss, bins2[1:], hist2, p0=p1_list)
        perr2 = np.sqrt(np.diag(pcov2))

        cut_th_peak_height = par1[1]
        cut_sep_height = par2[1]
        ss_eff = cut_th_peak_height / th_peak_height
        ms_eff = cut_sep_height / sep_peak_height
        cc_eff = cut_cc_height / cc_height
        ss_eff_array.append(ss_eff)
        ms_eff_array.append(ms_eff)
        cc_eff_array.append(cc_eff)
        cut_line_list.append(line)

        line += .001  ##<-- this can be lowered to get a more fine cut
                      ##at the cost of more computations

    print(line)

    ##Draw up 2D hist of AoverE vs E
    plt.hist2d(energy, y, bins=[1000,200], range=[[0, 2000], [0, 2]],
                norm=LogNorm(), cmap='jet')
    plt.hlines(line, 0, 2000, color='r', linewidth=1.5)
    cbar = plt.colorbar()
    plt.title("Dataset {}: 2D A/E vs E".format(ds.ds_list[0]))
    plt.xlabel("Energy (keV)", ha='right', x=1)
    plt.ylabel("A/Eunc", ha='right', y=1)
    cbar.ax.set_ylabel('Counts')
    plt.tight_layout()
    plt.show()

    ##1D hist of A/E and the cut line
    plt.clf()
    a1 = y[np.where((1589 < energy) & (energy < 1595))]
    hist, bins = np.histogram(a1, bins = 100, range=[.4,1.5])
    plt.vlines(line, 0, np.max(hist), color='r', linewidth=1.5)
    plt.plot(bins[1:], hist)
    plt.xlabel('A over E normalized')
    plt.ylabel('Counts')
    plt.title("Dataset {}: 1D A/E".format(ds.ds_list[0]))
    plt.show()

    plt.clf()
    plt.plot(cut_line_list, ss_eff_array, label='ss_eff')
    plt.plot(cut_line_list, ms_eff_array, label='ms_eff')
    plt.plot(cut_line_list, cc_eff_array, label='cc_eff')
    plt.ylabel('eff')
    plt.xlabel('AoverE_normalized')
    plt.title("Dataset {}: Efficiency curve".format(ds.ds_list[0]))
    plt.legend()
    plt.show()

    hist, bins = np.histogram(energy, bins=2000, range=[0,2000])
    hist1, bins1 = np.histogram(e1, bins=2000, range=[0,2000])

    plt.clf()
    plt.semilogy(bins[1:], hist, color='black', ds="steps", linewidth=1.5,
                label='Calibrated Energy: Dataset {}'.format(ds.ds_list[0]))
    plt.semilogy(bins1[1:], hist1, '-r', ds="steps", linewidth=1.5,
                label='AvsE Cut: Dataset {}'.format(ds.ds_list[0]))
    plt.ylabel('Counts')
    plt.xlabel('keV')
    plt.title("Dataset {}: Original Energy and Cut Energy".format(ds.ds_list[0]))
    plt.legend()
    plt.tight_layout()
    plt.show()

    return line, ss_eff_array, cc_eff_array, cut_line_list


def th_232(energy, a_over_e, ds, write_db=False):

    ## Find A/E cut for Th232 source

    #FWHM of nearby peaks is 2.5
    cc_range = [1790,2050]
    dep_range = [1530,1620]
    hist, bins = np.histogram(energy, bins=(dep_range[1]-dep_range[0]),
                              range=dep_range)
    b = (bins[:-1] + bins[1:]) / 2

    p0_list = [1588.2, 400, 2.5, 1592.5, 400, 2.5, 157]
    bnds = ([1588, 0, .9*p0_list[2], 1592.5, 0, .9*p0_list[5], 0],
            [1589, 700, 1.1*p0_list[2], 1593, 700, 1.1*p0_list[5], 300])

    par, pcov = curve_fit(gauss, b, hist, p0=p0_list, bounds=bnds)
    print(par)
    perr = np.sqrt(np.diag(pcov))
    print(perr)

    # np.savez('double_gauss_params', par)

    plt.title('Peak 1590 combined')
    plt.plot(b, hist, ds="steps", color='black')
    plt.plot(b, gauss(b, *par), '-r')
    plt.tight_layout()
    plt.show()

    cc_height = np.sum(energy[np.where((energy > cc_range[0]) & (energy < cc_range[1]))]) / (cc_range[1]-cc_range[0])

    ac_peak_height = par[1]
    th_peak_height = par[4]
    cut_ac_peak_height = par[1]
    cut_th_peak_height = par[4]
    ss_eff_array = []
    ms_eff_array = []
    cc_eff_array = []
    cut_line_list = []

    line = .8
    print("Finding optimal cut, keeping 90% of 1592 DEP")

    while cut_th_peak_height > .9 * th_peak_height:

        y = a_over_e[np.where(line < a_over_e)]
        e1 = energy[np.where(line < a_over_e)]
        cut_cc_height = np.sum(e1[np.where((e1 > cc_range[0]) & (e1 < cc_range[1]))]) / (cc_range[1]-cc_range[0])

        hist1, bins1 = np.histogram(e1, bins=(dep_range[1]-dep_range[0]),
                                    range=dep_range)

        par1, pcov1 = curve_fit(
            gauss, b, hist1, p0=p0_list, bounds=bnds)
        perr1 = np.sqrt(np.diag(pcov1))

        cut_ac_peak_height = par1[1]
        cut_th_peak_height = par1[4]
        ss_eff = cut_th_peak_height / th_peak_height
        ms_eff = cut_ac_peak_height / ac_peak_height
        cc_eff = cut_cc_height / cc_height
        ss_eff_array.append(ss_eff)
        ms_eff_array.append(ms_eff)
        cc_eff_array.append(cc_eff)
        cut_line_list.append(line)

        line += .001

    print(line)

    plt.clf()
    plt.hist2d(energy, a_over_e, bins=[1350,200], range=[[0, 2700], [0, 2]],
                norm=LogNorm(), cmap='jet')
    plt.hlines(line, 0, 2700, color='r', linewidth=1.5)
    plt.xlabel("Energy (keV)", ha='right', x=1)
    plt.ylabel("A/Eunc", ha='right', y=1)
    plt.title("Dataset {}: 2D A/E vs E".format(ds.ds_list[0]))
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts')
    plt.tight_layout()
    plt.show()

    plt.clf()
    a1 = a_over_e[np.where((1589 < energy) & (energy < 1595))]
    hist, bins = np.histogram(a1, bins = 100, range=[.4,1.5])
    plt.vlines(line, 0, np.max(hist), color='r', linewidth=1.5)
    plt.plot(bins[1:], hist)
    plt.title("Dataset {}: 1D A/E".format(ds.ds_list[0]))
    plt.xlabel('A over E normalized')
    plt.ylabel('Counts')
    plt.show()

    plt.clf()
    plt.plot(cut_line_list, ss_eff_array, label='ss_eff')
    plt.plot(cut_line_list, ms_eff_array, label='ms_eff')
    plt.plot(cut_line_list, cc_eff_array, label='cc_eff')
    plt.title("Dataset {}: Efficiency curves".format(ds.ds_list[0]))
    plt.ylabel('eff')
    plt.xlabel('AoverE_normalized')
    plt.legend()
    plt.show()

    hist, bins = np.histogram(energy, bins=2700, range=[0,2700])
    hist1, bins1 = np.histogram(e1, bins=2700, range=[0,2700])

    plt.clf()
    plt.semilogy(bins[1:], hist, color='black', ds="steps", linewidth=1.5,
                label='Calibrated Energy: Dataset {}'.format(ds.ds_list[0]))
    plt.semilogy(bins1[1:], hist1, '-r', ds="steps", linewidth=1.5,
                label='AvsE Cut: Dataset {}'.format(ds.ds_list[0]))
    plt.ylabel('Counts')
    plt.xlabel('keV')
    plt.title("Dataset {}: Original Energy and Cut Energy".format(ds.ds_list[0]))
    plt.legend()
    plt.tight_layout()
    plt.show()

    return line, ss_eff_array, ms_eff_array, cc_eff_array, cut_line_list


def linear_correction(energy, a_over_e):

    """
    Make a linear correction to A/E
    TODO: Use compt continuum bkg areas with a gaussian fit to improve linear correction
    """

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
    par, pcov = curve_fit(linear_func, peak_list, max_list)

    a_over_e = a_over_e / (par[0] * energy + par[1])

    return a_over_e

def gauss(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params) - 1, 3):
        x0 = params[i]
        a = params[i + 1]
        sigma = params[i + 2]
        y += a * np.exp(-(x - x0)**2 / (2 * sigma**2))
    y = y + params[-1]
    return y

def gauss_new(x, *params):
    #[th_peak_loc, th_amp, ac_amp, bkg]
    x0 = params[0]
    x1 = x0 - 4.5
    a = params[1]
    a1 = params[2]
    sigma = 2.75
    sigma1 = 2.5
    y = a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + a1*np.exp(-(x - x1)**2 / (2 * sigma1**2)) + params[3]
    return y

def linear_func(x,a,b):
    return a*x + b

## Below is a bunch of test code to try and optimize the cut and cut_algorithm
## DOES NOT WORK ATM

def test_code(a_over_e, energy, ds):
    """
    Function for testing a better algorithm to find the cut, currently in shambles
    and needs A LOT more work.
    """


    dep_range = [1530,1620]
    aoe_init_cut = a_over_e[np.where((energy > 1590) & (energy < 1596))]
    hist, bins = np.histogram(aoe_init_cut, bins=500, range=[.1,1.5])
    b = (bins[:-1] + bins[1:]) / 2


    p0_aoe = [1, 328, .006, 23]

    # p0_ac_dep = [1588.2, 400, 2.5, 1592.5, 400, 2.5, 157]
    # bnds = ([1588, 0, .9*p0_ac_dep[2], 1592.5, 0, .9*p0_ac_dep[5], 0],
    #         [1589, 700, 1.1*p0_ac_dep[2], 1593, 700, 1.1*p0_ac_dep[5], 300])
    p0_ac_dep = [1593, 400, 400, 157]
    bnds = ([1592, 0, 0, 0],
            [1593.5, 1000, 1000, 500])

    par, pcov = curve_fit(gauss, b, hist, p0=p0_aoe)
    print(par)
    perr = np.sqrt(np.diag(pcov))
    print(perr)

    init_line = par[0] - 2.4 * par[2]
    cut_step_size = .001 #* par[1] ##IMPORTANT
    total_steps = int(round((init_line - .8) / cut_step_size))

    aoe = a_over_e[np.where((energy > 1530) & (energy < 1620))]
    hist, bins = np.histogram(aoe, bins=500, range=[.1,1.5])
    b = (bins[:-1] + bins[1:]) / 2

    plt.plot(b, hist)
    plt.plot(b, gauss(b, *par), '-r')
    plt.vlines(init_line, 0, 300, color='r', linewidth=1.5)
    plt.xlabel("AoE Normalized", ha='right', x=1)
    plt.ylabel("Counts", ha='right', y=1)
    plt.tight_layout()
    plt.show()


    e_hist, bins = np.histogram(energy, bins=(dep_range[1]-dep_range[0]), range=dep_range)
    b = (bins[:-1] + bins[1:]) / 2

    par, pcov = curve_fit(gauss_new, b, e_hist, p0=p0_ac_dep, bounds=bnds)
    print(par)
    perr = np.sqrt(np.diag(pcov))
    print(perr)

    ac_peak_height = par[2]
    th_peak_height = par[1]

    plt.clf()
    plt.title('Peak 1590 combined')
    plt.plot(b, e_hist, ls="steps", color='black')
    plt.plot(b, gauss_new(b, *par), '-r')
    plt.tight_layout()
    plt.show()


    line = init_line



    cut_params = {
        "ac_peak_height":par[2],
        "th_peak_height": par[1],
        "ac_error": perr[2],
        "th_error": perr[1],
        "dep_range": [1530,1620],
        "cut_step_size": cut_step_size,
        "total_steps": total_steps
        }


    cut_algorithm(energy, a_over_e, line, cut_params)


    """
    for i in range(20):

        e1 = energy[np.where(line < a_over_e)]
        hist1, bins1 = np.histogram(e1, bins=(dep_range[1]-dep_range[0]), range=dep_range)
        b = (bins1[:-1] + bins1[1:]) / 2

        par, pcov = curve_fit(gauss, b, hist1, p0=p0_ac_dep, bounds=bnds)
        p_error = np.sqrt(np.diag(pcov))


        cut_ac_peak_height = par[1]
        cut_th_peak_height = par[4]
        ss_eff = cut_th_peak_height / th_peak_height
        ms_eff = cut_ac_peak_height / ac_peak_height
        ss_eff_array.append(ss_eff)
        ms_eff_array.append(ms_eff)
        cut_line_list.append(line)

        line -= cut_step_size
    """

    # plt.clf()
    # plt.hist2d(energy, a_over_e, bins=[1000,200], range=[[0, 2000], [0, 2]], norm=LogNorm(), cmap='jet')
    # plt.hlines(line, 0, 2000, color='r', linewidth=1.5)
    # plt.xlabel("Energy (keV)", ha='right', x=1)
    # plt.ylabel("A/Eunc", ha='right', y=1)
    # cbar = plt.colorbar()
    # cbar.ax.set_ylabel('Counts')
    # plt.tight_layout()
    # plt.show()

    # plt.clf()
    # a1 = a_over_e[np.where((1589 < energy) & (energy < 1595))]
    # hist, bins = np.histogram(a_over_e, bins = 200, range=[.4,1.5])
    # plt.vlines(line, 0, 300000, color='r', linewidth=1.5)
    # plt.plot(bins[1:], hist)
    # plt.xlabel('A over E normalized')
    # plt.ylabel('Counts')
    # plt.show()

    plt.clf()
    plt.scatter(cut_line_list, ss_eff_array, label='ss_eff')
    # plt.plot(cut_line_list, ms_eff_array, label='ms_eff')
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



    # plt.hist2d(energy, a_over_e, bins=[1000,200], range=[[0, 2000], [0, 2]], norm=LogNorm(), cmap='jet')
    # plt.hlines(line, 0, 2000, color='r', linewidth=1.5)
    # plt.xlabel("Energy (keV)", ha='right', x=1)
    # plt.ylabel("A/Eunc", ha='right', y=1)
    # cbar = plt.colorbar()
    # cbar.ax.set_ylabel('Counts')
    # plt.tight_layout()
    # plt.show()

def cut_algorithm(energy, a_over_e, line, cut_params):

    """
    Heart of algorithm for test code, needs A LOT more work
    """

    ss_eff_array = []
    ms_eff_array = []
    cut_line_list = []
    cut_step_size = cut_params['cut_step_size']
    cut_height_list = []
    cut_error_up = []
    cut_error_low = []
    p0_ac_dep = [1593, 400, 200, 157]
    bnds = ([1592, 0, 0, 0],
            [1593.5, 1000, 1000, 500])


    # for i in range(40):
    for i in range(cut_params['total_steps']):

        e1 = energy[np.where(line < a_over_e)]
        hist, bins = np.histogram(energy, bins=(cut_params['dep_range'][1]-cut_params['dep_range'][0]), range=cut_params['dep_range'])
        hist1, bins1 = np.histogram(e1, bins=(cut_params['dep_range'][1]-cut_params['dep_range'][0]), range=cut_params['dep_range'])
        b = (bins1[:-1] + bins1[1:]) / 2

        par, pcov = curve_fit(gauss_new, b, hist1, p0=p0_ac_dep, bounds=bnds)
        p_error = np.sqrt(np.diag(pcov))


        # cut_ac_peak_height = par[2]
        cut_th_peak_height = par[1]
        # cut_error.append(p_error[4])
        ss_eff = cut_th_peak_height / cut_params['th_peak_height']
        # ms_eff = cut_ac_peak_height / cut_params['ac_peak_height']
        ci_low, ci_upp = proportion.proportion_confint(cut_th_peak_height, cut_params['th_peak_height'], alpha=0.05, method='beta')
        cut_height_list.append(cut_th_peak_height)
        ss_eff_array.append(ss_eff)
        # ms_eff_array.append(ms_eff)
        cut_line_list.append(line)
        cut_error_up.append(ci_upp)
        cut_error_low.append(ci_low)

        line -= cut_step_size

    errors = [cut_error_low, cut_error_up]
    cut_height_list = np.asarray(cut_height_list)
    ss_eff_array = np.asarray(ss_eff_array)
    cut_line_list = np.asarray(cut_line_list)


    # errors = ss_eff_array * np.sqrt((cut_error/cut_height_list)**2)
    # errors = ss_eff_array * cut_error/cut_height_list


    idx = (np.abs(ss_eff_array - .9)).argmin()
    ss_eff_array = ss_eff_array[(idx-5):(idx+5)]
    cut_line_list = cut_line_list[(idx-5):(idx+5)]
    cut_error_low = cut_error_low[(idx-5):(idx+5)]
    cut_error_up = cut_error_up[(idx-5):(idx+5)]
    errors = [cut_error_low, cut_error_up]


    params, pcovariance = curve_fit(linear_func, cut_line_list, ss_eff_array)
    param_error = np.sqrt(np.diag(pcovariance))

    cut = (.9 - params[1]) / params[0]
    print("Here is the cut from linear fit: ", cut)

    plt.clf()
    plt.scatter(cut_line_list, ss_eff_array, s=20, label='ss_eff')
    plt.plot(cut_line_list, linear_func(cut_line_list, *params), '-r')
    # plt.errorbar(cut_line_list, ss_eff_array, yerr=errors, fmt='o', markersize='4', label='ss_eff')
    # # plt.plot(cut_line_list, ms_eff_array, label='ms_eff')
    plt.ylabel('eff')
    plt.xlabel('AoverE_normalized_cut')
    plt.legend()
    plt.show()

    e1 = energy[np.where(cut < a_over_e)]
    par, pcov = curve_fit(gauss_new, b, hist1, p0=p0_ac_dep, bounds=bnds)
    p_error = np.sqrt(np.diag(pcov))
    print(par)
    hist, bins = np.histogram(energy, bins=2600, range=[0,2600])
    hist1, bins1 = np.histogram(e1, bins=2600, range=[0,2600])

    plt.clf()
    plt.semilogy(bins[1:], hist, color='black', ls="steps", linewidth=1.5)
    plt.semilogy(bins1[1:], hist1, '-r', ls="steps", linewidth=1.5)
    plt.semilogy(bins1[1:], gauss_new(bins1[1:], *par), color='blue', linewidth=1.5)
    plt.ylabel('Counts')
    plt.xlabel('keV')
    plt.tight_layout()
    plt.show()

    exit()



if __name__=="__main__":
    main()
