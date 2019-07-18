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
    global display
    display = 1 # allow displaying intermediate distributions for control
    
    run_db, cal_db = "runDB.json", "calDB.json"

    with open(run_db) as f:
        runDB = json.load(f)

    global tier_dir
    tier_dir = runDB["tier_dir"]
    global meta_dir
    meta_dir = runDB["meta_dir"]
    global dep_line
    dep_line = 1592.5
    global dep_acc
    dep_acc = 0.9

    peaks_of_interest = sorted(runDB["peaks_of_interest"], reverse=True)

    # take calibration parameter for the 'calibration.py' output
    with open(cal_db) as f:
      calDB = json.load(f)

    par = argparse.ArgumentParser(description="calibration suite for tumbsi")
    arg, st, sf = par.add_argument, "store_true", "store_false"
    arg("-ds", nargs='*', action="store", help="load runs for a DS")
    arg("-r", "--run", nargs=1, help="load a single run")
    args = vars(par.parse_args())

    ecal = np.zeros(3)
    ecal[0] = calDB["cal_pass1"]["1"]["p1cal"]
    if("cal_pass2" in calDB):
      ecal[1] = calDB["cal_pass2"]["1"]["p2acal"]
      ecal[2] = calDB["cal_pass2"]["1"]["p2bcal"]

    eres = np.zeros(2)
    if("eres_curve" in calDB):
      eres[0] = calDB["eres_curve"]["1"]["acal"]
      eres[1] = calDB["eres_curve"]["1"]["bcal"]
    else:
      print("You must run a calibration to get the energy resolution curve. Exit.")
      sys.exit()

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

    print("")
    print("Start Pulse Shape Anlysis")
    print("")

    psa(run, ds, ecal, eres, peaks_of_interest)

def psa(run, dataset, ecal, eres, peaks_of_interest):

    # ds = DataSet(runlist=[191], md='./runDB.json', tier_dir=tier_dir)
    ds = DataSet(ds_lo=0, md='./runDB.json', tier_dir=tier_dir)
    t2 = ds.get_t2df()
    # t2df = os.path.expandvars('{}/Spectrum_{}.hdf5'.format(meta_dir,run))
    # t2df = pd.read_hdf(t2df, key="df")
    # t2df = t2df.reset_index(drop=True)
    t2 = t2.reset_index(drop=True)
    print("  Energy calibration:")
    cal = ecal[0] * np.asarray(t2["e_ftp"])
    print("  -> 1st pass linear energy calibration done")
    if(cal[1]):
      cal = cal/ecal[1] - ecal[2]
      print("  -> 2nd pass linear energy calibration done")

    n = "current_max"
    e = "e_cal"
    e_over_unc = cal / np.asarray(t2["e_ftp"])
    aoe0 = np.asarray(t2[n])

    print("  Apply quality cuts")
    Nall = len(cal)

    bl0 = np.asarray(t2["bl0"])
    bl1 = np.asarray(t2["bl1"])
    e_over_unc = e_over_unc[(bl1-bl0)<2]
    cal = cal[(bl1-bl0)<2]
    aoe0 = aoe0[(bl1-bl0)<2]

    Nqc_acc = len(cal)

    print("  -> Total number of events: ",Nall)
    print("  -> After quality cuts    : ",Nqc_acc)
    print("  -> Quality cuts rejection: ",100*float(Nqc_acc)/float(Nall),"%")

    aoe = aoe0 * e_over_unc / cal

    print("  Compute AoE normalization curve")
    aoe_norm = AoEcorrection(cal,aoe)
    print("  -> parameteres (a x E + b):",aoe_norm[0],aoe_norm[1])
    aoe = aoe/(aoe_norm[0]*cal + aoe_norm[1])

    print("  Find the low-side A/E cut for ",100*dep_acc,"% 208Tl DEP acceptance")
    cut = get_aoe_cut(cal,aoe,dep_line,eres)
    print("  -> cut: ",'{:1.3f}'.format(cut))
    if cut==0:
      print("  -> cut not found. Exit.")
      sys.exit()

    print("  Compute energy spectrum after A/E cut")
    cal_cut = cal[aoe>=cut]

    print("  Compute survival fractions: ")
    sf = np.zeros(len(peaks_of_interest))
    sferr = np.zeros(len(peaks_of_interest))
    for i,peak in enumerate(peaks_of_interest):
      sf[i], sferr[i] = get_sf(cal,aoe,cut,peak,eres)
      print("  -> ",peak,'{:2.1f}'.format(100.*sf[i])," +/- ",'{:2.1f}'.format(100.*sferr[i]),"%")

    print("  Display hitograms")
    plt.figure(2)
    plt.hist2d(cal, aoe, bins=[2000,400], range=[[0, 3000], [0, 1.5]], norm=LogNorm(), cmap='jet')
    cbar = plt.colorbar()
    plt.title("Dataset {}".format(dataset))
    plt.xlabel("Energy (keV)", ha='right', x=1)
    plt.ylabel("A/E (a.u.)", ha='right', y=1)
    cbar.ax.set_ylabel('Counts')
    plt.tight_layout()
    plt.savefig('./plots/aoe_versus_energy.pdf', bbox_inches='tight', transparent=True)
    plt.show()

    plt.figure(3)
    hist, bins = np.histogram(cal, bins=3000, range=[0,3000])
    hist1, bins1 = np.histogram(cal_cut, bins=3000, range=[0,3000])
    plt.clf()
    plt.plot(bins[1:], hist, color='black', ls="steps", linewidth=1.5, label='all events')
    plt.plot(bins1[1:], hist1, '-r', ls="steps", linewidth=1.5, label='after A/E cut')
    plt.ylabel('Counts', ha='right', y=1)
    plt.xlabel('Energy (keV)', ha='right', x=1)
    plt.legend(title='Calibrated Energy')
    plt.yscale('log')
    plt.savefig('./plots/calEnergy_spectrum_after_psa.pdf', bbox_inches='tight', transparent=True)
    plt.show()

    print("")
    print("  Normal termination")
    print("")

def get_aoe_cut(energy,aoe,peak,eres):
    sf = 0
    fwhm = np.sqrt(eres[0]*peak+eres[1])
    
    # set side bands and peak boundaries
    emin_low_band  = peak - 4*fwhm
    emax_low_band  = peak - 2*fwhm
    emin           = peak - 2*fwhm
    emax           = peak + 2*fwhm
    emin_high_band = peak + 2*fwhm
    emax_high_band = peak + 4*fwhm
    
    # define the A/E distrbutions
    aoe_bkg = [[],[],[]]
    aoe_bkg[0] = aoe[(energy>emin_low_band) & (energy<emax_low_band)]
    aoe_bkg[1] = aoe[(energy>emin) & (energy<emax)]
    aoe_bkg[2] = aoe[(energy>emin_high_band) & (energy<emax_high_band)]
    
    hist_bkg = np.zeros(500)
    hist_sig = np.zeros(500)
    bin_center = np.zeros(500)
    
    if display: plt.figure(1)
    # fill in bkg and signal histograms
    for i,c in zip(range(len(aoe_bkg)),mcolors.TABLEAU_COLORS):
      hist, bins = np.histogram(aoe_bkg[i],bins=500,range=[0,1.5])
      bin_center = (bins[:-1] + bins[1:]) / 2
      
      if i == 0: hist_bkg = hist
      elif i == 1: hist_sig = hist
      else: hist_bkg = [x + y for x, y in zip(hist, hist_bkg)]

    # perform the bkg subtraction
    hist_sig = [x - y for x, y in zip(hist_sig, hist_bkg)]

    cut = 0
    
    Ntot = sum(hist_sig)
    for bin in bin_center:
      sf = float(sum(np.array(hist_sig)[bin_center > bin]))/float(Ntot)
      if sf > dep_acc:
        low_bin = bin
        low_sf = sf
      if sf < dep_acc:
        high_bin = bin
        high_sf = sf
        break

    cut = low_bin + (high_bin-low_bin)/(high_sf-low_sf)*(dep_acc-low_sf)

    return cut

def get_sf(energy,aoe,cut,peak,eres):
    
    sf = 0
    fwhm = np.sqrt(eres[0]*peak+eres[1])

    # set side bands and peak boundaries
    emin_low_band  = peak - 4*fwhm
    emax_low_band  = peak - 2*fwhm
    emin           = peak - 2*fwhm
    emax           = peak + 2*fwhm
    emin_high_band = peak + 2*fwhm
    emax_high_band = peak + 4*fwhm
    
    # define the A/E distrbutions
    aoe_bkg = [[],[],[]]
    aoe_bkg[0] = aoe[(energy>emin_low_band) & (energy<emax_low_band)]
    aoe_bkg[1] = aoe[(energy>emin) & (energy<emax)]
    aoe_bkg[2] = aoe[(energy>emin_high_band) & (energy<emax_high_band)]

    hist_bkg = np.zeros(500)
    hist_sig = np.zeros(500)
    bin_center = np.zeros(500)
    
    if display: plt.figure(1)
    # perform the side bands subtraction
    for i,c in zip(range(len(aoe_bkg)),mcolors.TABLEAU_COLORS):
      hist, bins = np.histogram(aoe_bkg[i],bins=500,range=[0,1.5])
      bin_center = (bins[:-1] + bins[1:]) / 2
    
      if i == 0:
        hist_bkg = hist
        label = 'low side band'
      elif i == 1:
        hist_sig = hist
        label = str('{:4.1f}'.format(peak)) + ' +/- ' + str('{:2.1f}'.format(2*fwhm)) + ' keV';
      else:
        hist_bkg = [x + y for x, y in zip(hist, hist_bkg)]
        label = 'high side band'
      
      if display:
        plt.subplot(211)
        axes = plt.gca()
        axes.set_xlim([0,1.5])
        plt.plot(bin_center,hist, color=c, label=label)
        plt.xlabel("A/E (a.u.)", ha='right', x=1)
        plt.ylabel("counts ", ha='right', y=1)
        plt.legend()

    if peak != 2039: hist_sig = [x - y for x, y in zip(hist_sig, hist_bkg)]
    
    if display:
      plt.subplot(212)
      axes = plt.gca()
      axes.set_xlim([0,1.5])
      plt.plot(bin_center,hist_sig,color='black',label='Bkg subtracted distribution')
      plt.legend()
      plt.xlabel("A/E (a.u.)", ha='right', x=1)
      plt.ylabel("bkg subtracted counts ", ha='right', y=1)
      plt.show()

    # compute the survival fraction given the cut value
    Ntot = sum(hist_sig)
    Nacc = sum(np.array(hist_sig)[bin_center > cut])
    if Nacc < 0: Nacc=0
  
    return float(Nacc)/float(Ntot), 1./float(Ntot)*np.sqrt(float(Nacc)+float(Nacc)*float(Nacc)/float(Ntot))

def AoEcorrection(e,aoe):

  comptBands_width = 20;
  comptBands = np.array([1000,1020,1040,1130,1150,1170,1190,1210,1250,1270,1290,1310,1330,1420,1520,1540,1700,1780,1810,1850,1870,1890,1910,1930,1950,1970,1990,2010,2030,2050,2150])
  compt_aoe = np.zeros(len(comptBands))
  
  for i, band in enumerate(comptBands):
    aoe_tmp = aoe[(e>band) & (e<band+comptBands_width) & (aoe>0.001)]
  
    hist, bins = np.histogram(aoe_tmp,bins=500)
    bin_center = (bins[:-1] + bins[1:]) / 2
    
    def gauss(x, mu, sigma, A):
      return A * (1. / sigma / np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2. * sigma**2))
    
    pars, cov = curve_fit(gauss, bin_center, hist, p0=[bins[np.argmax(hist)],1e-3,np.max(hist)])
    compt_aoe[i] = pars[0]

    # display A/E histogram fit
    if display:
      plt.figure(4)
      model = np.zeros(len(bin_center))
      for i,bin in enumerate(bin_center):
        model[i] = gauss(bin,pars[0],pars[1],pars[2])
    
      plt.plot(bin_center,hist)
      plt.plot(bin_center,model,'-',color='red',label='Compton region: ' + str(band) + ' keV')
      plt.legend()
      plt.xlabel("raw A/E (a.u.)", ha='right', x=1)
      plt.ylabel("counts ", ha='right', y=1)
      plt.show()

  def pol1(x,a,b):
    return a * x + b

  pars, cov = opt.curve_fit(pol1,comptBands,compt_aoe)
  errs = np.sqrt(np.diag(cov))

  model = np.zeros(len(comptBands))
  for i,bin in enumerate(comptBands):
        model[i] = pol1(bin,pars[0],pars[1])
  
  # display linear A/E versus E fit
  if display:
    plt.figure(5)
    plt.plot(comptBands,compt_aoe,label='data')
    plt.plot(comptBands,model,label='linear model')
    plt.legend(title='A/E energy dependence')
    plt.xlabel("Energy (keV)", ha='right', x=1)
    plt.ylabel("raw A/E (a.u.)", ha='right', y=1)
    plt.savefig('./plots/aoe_energy_dependence.pdf', bbox_inches='tight', transparent=True)
    plt.show()

  return pars

if __name__=="__main__":
    main()
