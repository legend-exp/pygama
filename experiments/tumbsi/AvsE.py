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
import scipy.optimize as opt
import tinydb as db
import argparse
import iminuit as im

from pygama import DataSet
from pygama.analysis.calibration import *
from pygama.analysis.histograms import *
import pygama.analysis.peak_fitting as pga
import pygama.utils as pgu
from matplotlib.lines import Line2D
from pygama.utils import set_plot_style

def main():
    """
    tumbsi analysis suite
    """
    run_db, cal_db = "runDB.json", "calDB.json"

    with open(run_db) as f:
        runDB = json.load(f)

    global tier_dir
    tier_dir = runDB["tier_dir"]
    global meta_dir
    meta_dir = runDB["meta_dir"]

    # take calibration parameter for the 'calibration.py' output
    with open(cal_db) as f:
      calDB = json.load(f)

    par = argparse.ArgumentParser(description="calibration suite for tumbsi")
    arg, st, sf = par.add_argument, "store_true", "store_false"
    arg("-ds", nargs='*', action="store", help="load runs for a DS")
    arg("-r", "--run", nargs=1, help="load a single run")
    args = vars(par.parse_args())

    cal1a = calDB["cal_pass1"]["1"]["p1cal"]
    cal2a = 0
    cal2b = 0
    if("cal_pass2" in calDB):
      cal2a = calDB["cal_pass2"]["1"]["p2acal"]
      cal2b = calDB["cal_pass2"]["1"]["p2bcal"]
    
    # Which run number is the being analyzed
    if args["ds"]:
      ds_lo = int(args["ds"][0])
      try:
        ds_hi = int(args["ds"][1])
      except:
        ds_hi = None
      ds = DataSet(ds_lo, ds_hi,md=run_db, cal=cal_db)
      run = ds_lo
  
    if args["run"]:
      ds = DataSet(run=int(args["run"][0]),md=run_db, cal=cal_db)

    histograms(run, ds, cal1a, cal2a, cal2b)

def histograms(run, dataset, cal1a, cal2a, cal2b):

    # ds = DataSet(runlist=[191], md='./runDB.json', tier_dir=tier_dir)
    ds = DataSet(ds_lo=0, md='./runDB.json', tier_dir=tier_dir)
    t2 = ds.get_t2df()
    # t2df = os.path.expandvars('{}/Spectrum_{}.hdf5'.format(meta_dir,run))
    # t2df = pd.read_hdf(t2df, key="df")
    # t2df = t2df.reset_index(drop=True)
    t2 = t2.reset_index(drop=True)
    cal = cal1a * np.asarray(t2["e_ftp"])
    print("1st pass linear energy calibration done")
    if(cal2a):
      cal = cal/cal2a - cal2b
      print("2nd pass linear energy calibration done")

    n = "current_max"
    e = "e_cal"
    e_over_unc = cal / np.asarray(t2["e_ftp"])
    # x = e_cal
    y0 = np.asarray(t2[n])
    y = y0 * e_over_unc / cal

    aoe_norm = AoEcorrection(cal,y)
    print("AoE normalization curve: ",aoe_norm[0],aoe_norm[1])
    y = y/(aoe_norm[0]*cal + aoe_norm[1])

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

    p0_list = [1592.5, 200, 3, 4]
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
    line = 0.8

    y1 = y[np.where(line < y)]
    x1 = cal[np.where(line < y)]
    # hist1, bins1 = np.histogram(x1, bins=500, range=[1500,1700])
    hist1, bins1 = np.histogram(x1, bins=450, range=[1530,1620])
    hist1 = hist1*5

    while cut > 0.9 * answer:

      #print(" -> ",line,0.9 * answer,cut)
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

        line += .0001

    print(line, cut)
    plt.hist2d(cal, y, bins=[2000,400], range=[[0, 3000], [0, 1.5]], norm=LogNorm(), cmap='jet')
    plt.hlines(line, 0, 3000, color='r', linewidth=1.5)
    cbar = plt.colorbar()
    plt.title("Dataset {}".format(dataset))
    plt.xlabel("Energy (keV)", ha='right', x=1)
    plt.ylabel("A/Eunc", ha='right', y=1)
    cbar.ax.set_ylabel('Counts')
    plt.tight_layout()
    plt.show()

    hist, bins = np.histogram(cal, bins=3000, range=[0,3000])
    hist1, bins1 = np.histogram(x1, bins=3000, range=[0,3000])
    plt.clf()
    plt.plot(bins[1:], hist, color='black', ls="steps", linewidth=1.5, label='Calibrated Energy: Dataset {}'.format(dataset))
    plt.plot(bins1[1:], hist1, '-r', ls="steps", linewidth=1.5, label='AvsE Cut: Dataset {}'.format(dataset))
    # plt.plot(bins[1:], gauss(bins[1:], *par), color='blue')
    # plt.plot(bins1[1:], gauss(bins1[1:], *par1), color='aqua')
    plt.ylabel('Counts')
    plt.xlabel('keV')
    plt.legend()
    plt.tight_layout()
    plt.yscale('log')
    plt.show()

def AoEcorrection(e,aoe):
  print("Start AoE normalization: ")

  comptBands_width = 20;
  comptBands = np.array([1000,1020,1040,1130,1150,1170,1190,1210,1250,1270,1290,1310,1330,1420,1520,1540,1700,1780,1810,1850,1870,1890,1910,1930,1950,1970,1990,2010,2030,2050,2150])
  compt_aoe = np.zeros(len(comptBands))
  
  for i, band in enumerate(comptBands):
    aoe_tmp = aoe[np.logical_and(np.logical_and(e>band,e<band+comptBands_width),aoe>0.001)]

    hist, bins = np.histogram(aoe_tmp,bins=500)
    bin_center = (bins[:-1] + bins[1:]) / 2
    
    def gauss(x, mu, sigma, A):
      return A * (1. / sigma / np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2. * sigma**2))
    
    pars, cov = curve_fit(gauss, bin_center, hist, p0=[bins[np.argmax(hist)],1e-3,np.max(hist)])
    compt_aoe[i] = pars[0]

    # disply A/E histogram fit
    #    model = np.zeros(len(bin_center))
    #    for i,bin in enumerate(bin_center):
    #      model[i] = gauss(bin,pars[0],pars[1],pars[2])
    
    #    plt.plot(bin_center,hist)
    #    plt.plot(bin_center,model,'-',color='red')
    #    plt.show()

  def pol1(x,a,b):
    return a * x + b

  pars, cov = opt.curve_fit(pol1,comptBands,compt_aoe)
  errs = np.sqrt(np.diag(cov))

  model = np.zeros(len(comptBands))
  for i,bin in enumerate(comptBands):
        model[i] = pol1(bin,pars[0],pars[1])
  
  # display linear A/E versus E fit
  #plt.plot(comptBands,compt_aoe)
  #plt.plot(comptBands,model)
  #plt.show()

  return pars

if __name__=="__main__":
    main()
