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
    Code for varying bias runs: 1174-1176
    """


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

    resolution(ds, args["writeDB"])

def resolution(ds, write_db=False):

    t2 = ds.get_t2df()
    t2 = t2.reset_index(drop=True)
    run = ds.runs[0]


    # par = 4.43623e-01 #1400V
    # par = 4.23296e-01 #1500V
    par = 4.23296e-01 #1600V

    cal = par * np.asarray(t2['e_ftp'])

    def gauss(x, *params):
        y = np.zeros_like(x)
        for i in range(0, len(params) - 1, 3):
            x0 = params[i]
            a = params[i + 1]
            sigma = params[i + 2]
            y += a * np.exp(-(x - x0)**2 / (2 * sigma**2))
        y = y + params[-1]
        return y


    elo = 1400
    ehi = 1520
    hist, bins = np.histogram(cal, bins=(ehi-elo), range=[elo,ehi])
    b = (bins[:-1] + bins[1:]) / 2

    p0_list = [1460, 127, 5, 4]

    bnds = ([1450, 0, 0, 0],
            [1470, 500, 20, 100])

    par, pcov = curve_fit(
        gauss, b, hist, p0=p0_list, bounds=bnds)
    print(par)
    perr = np.sqrt(np.diag(pcov))
    print(perr)

    fwhm = 2*np.sqrt(2*np.log(2))*par[2]

    plt.title('Run {}: 1460 keV Peak Resolution'.format(run))
    plt.plot(b, hist, ls="steps", color='black')
    plt.plot(b, gauss(b, *par), '-r', label='fwhm={}'.format(fwhm))
    plt.legend()
    plt.tight_layout()
    plt.show()







if __name__=="__main__":
    main()
