#!/usr/bin/env python3.7
import argparse
import json
import sys
import os
from decimal import Decimal
import numpy as np
import pandas as pd
import tinydb as db
import matplotlib.pyplot as plt
import itertools as it
from scipy.stats import mode
import scipy.optimize as opt
from pprint import pprint
from pygama import DataSet
import pygama.utils as pgu
import pygama.analysis.histograms as pgh
import pygama.analysis.peak_fitting as pga

def main():
    """
    perform automatic calibration of pygama DataSets.
    command line options to specify the DataSet are the same as in processing.py
    save results in a JSON database for access by other routines.
    """

    par = argparse.ArgumentParser(description="calibration suite for tumbsi")
    arg, st, sf = par.add_argument, "store_true", "store_false"
    arg("-ds", nargs='*', action="store", help="load runs for a DS")
    arg("-r", "--run", nargs=1, help="load a single run")
    arg("-s", "--spec", action=st, help="print simple spectrum")
    arg("-sc", "--cal", action=st, help="print calibrated spectrum")
    arg("-p0","--pass0",action=st,help="run pass0 (single peak) calibration")
    arg("-p1", "--pass1", action=st, help="run pass-1 (linear) calibration")
    arg("-p2", "--pass2", action=st, help="run pass-2 (peakfit) calibration")
    arg("-e", "--etype", nargs=1, help="custom energy param (default is e_ftp)")
    arg("-t", "--test", action=st, help="set verbose (testing) output")
    arg("-w", "--writeDB", action=st, help="store results in DB")
    arg("-pr", "--printDB", action=st, help="print calibration results in DB")
    arg("-pa","--path",nargs=1,help="Set Path to runDB.json file")
    arg("-db","--db",nargs=1,help="Path to runDB.json and calDB.json")
    arg("-sub","--sub",nargs=1,help="Number of Subfiles")
    args = vars(par.parse_args())




    etype = args["etype"][0] if args["etype"] else "e_ftp"

    if args["db"]:
        path_to_files = args["db"][0]
        run_db, cal_db = path_to_files+"/runDB.json", path_to_files+"/calDB.json"
    # -- declare the DataSet --
    if args["ds"]:
        ds_lo = int(args["ds"][0])
        try:
            ds_hi = int(args["ds"][1])
        except:
            ds_hi = None
        ds = DataSet(ds_lo, ds_hi,1,
                     md=run_db, cal=cal_db, v=args["test"])

    if args["run"]:
        run = int(args["run"][0])
        ds = DataSet(1,run,
                     md=run_db, cal=cal_db, v=args["test"])
   
        fp = ds.paths[run]["t2_path"].split("t2")[0]
        t2_file=ds.get_t2df()
        if args["sub"]:
            subNumber = int(args["sub"][0])
            counter = 0
            for p, d, files in os.walk(ds.tier2_dir):
              for f in files:
                if any("{}-".format(r) in f for r in [run]):
                    if counter < subNumber:
                        t2_file = t2_file.append(pd.read_hdf(fp+f))
                counter += 1

    print("Whaat")

    if args["spec"]:
        his = t2_file.hist("e_ftp", bins = 2000)
        plt.yscale('log')
        plt.savefig(path_to_files+'plots/Raw.png',bbox_inches='tight',transperent=True)
        plt.show()
    
    if args["pass0"]:
        calibrate_pass0(ds, t2_file,etype,  args["writeDB"])
    if args["pass1"]:
        calibrate_pass1(ds, t2_file,etype,  args["writeDB"], args["test"])
  
    if args["pass2"]:
        calibrate_pass2(ds,t2_file, run, cal_db, run_db,args["writeDB"])

    if args["printDB"]:
        show_calDB(cal_db)
    
    if args["cal"]:
         show_calspectrum(ds,t2_file, cal_db,etype,run, args["pass1"],args["pass2"])



    # -- declare the DataSet --


def calibrate_pass0(ds, df, etype="e_ftp", write_db=False, test=False):

    print("Calibration on single peak")
 

    epeaks = sorted(ds.config["cal_peaks"], reverse=True)
    pu = epeaks[0]
    # get initial parameters for this energy estimator
    calpars = ds.get_p1cal_pars("e_ftp")
    pk_thresh = calpars["peakdet_thresh"]
    match_thresh = calpars["match_thresh"]
    xlo, xhi, xpb = calpars["xlims"]
    ene = df[etype]
    nb = int((xhi-xlo)/xpb)
    h, bins = np.histogram(ene, nb, (xlo, xhi))
    b = (bins[:-1] + bins[1:]) / 2.

    # run peakdet to identify the uncalibrated maxima
    maxes, mins = pgu.peakdet(h, pk_thresh, b)
    calVal = maxes[-1][0]
    fcal = pu/calVal
    firstcal = ene * fcal
    h1, bins1 = np.histogram(firstcal,1600,(1400,3000))
    b1 = (bins1[:-1] + bins1[1:]) / 2.
    maxes1, mins1 = pgu.peakdet(h1, pk_thresh, b1)
    print(maxes1)    
#    plt.plot(h1)
#    plt.show()
    iter = 0
    if len(maxes1) == len(epeaks):
       peaks = sorted(maxes1[:,0], reverse=True)

    else:
       print("found more or less peaks then expected! Have a look at the raw spectrum again...")
       peaks = maxes1[:,0]

    peaks = np.array(peaks,dtype=float)
    print("Peaks = " , peaks)
    def pol1(x,a,b):
      return a * x + b

    pars1, cov1 = opt.curve_fit(pol1,peaks,epeaks)
    errs1 = np.sqrt(np.diag(cov1))
    print("Calibration curve: ",pars1)
    
    ecal = firstcal*pars1[0]+pars1[1]
    hcal, bins2 = np.histogram(ecal,3000,(0,3000))
    plt.plot(hcal)
    plt.show()


def calibrate_pass1(ds, df, etype="e_ftp", write_db=False, test=False):
    """
    Run a "first guess" calibration of an arbitrary energy estimator.

    Uses a peak matching algorithm based on finding ratios of uncalibrated
    and "true" (keV) energies.  We run peakdet to find the maxima
    in the spectrum, then compute all ratios e1/e2, u1/u2, ..., u29/u30 etc.
    We find the subset of uncalibrated ratios (u7/u8, ... etc) that
    match the "true" ratios, and compute a calibration constant for each.
    Then for each uncalibrated ratio, we assume it to be true, then loop
    over the expected peak positions.
    We shift the uncalibrated peaks so that the true peak would be very close
    to 0, and calculate its distance from 0.  The "true" calibration constant
    will minimize this value for all ratios, and this is the one we select.

    Writes first-pass calibration results to a database, for access
    by the second pass, and other codes.
    """

    print("Start first Calibration!")
    # get a list of peaks we assume are always present
    epeaks = sorted(ds.config["cal_peaks"], reverse=True)

    # get initial parameters for this energy estimator
    calpars = ds.get_p1cal_pars(etype)
    pk_thresh = calpars["peakdet_thresh"]
    match_thresh = calpars["match_thresh"]
    xlo, xhi, xpb = calpars["xlims"]

    # make energy histogram
    ene = df[etype]
    nb = int((xhi-xlo)/xpb)
    h, bins = np.histogram(ene, nb, (xlo, xhi))
    b = (bins[:-1] + bins[1:]) / 2.
    print(pk_thresh, " / ", b)
    print("")
    print(bins)
    plt.plot(h)
    plt.show()
    # run peakdet to identify the uncalibrated maxima
    maxes, mins = pgu.peakdet(h, pk_thresh, b)
    print(maxes)
   
    umaxes = np.array(sorted([x[0] for x in maxes], reverse=True))

    # compute all ratios
    ecom = [c for c in it.combinations(epeaks, 2)]
    ucom = [c for c in it.combinations(umaxes, 2)]
    eratios = np.array([x[0] / x[1] for x in ecom]) # assumes x[0] > x[1]
    uratios = np.array([x[0] / x[1] for x in ucom])
    # match peaks to true energies
    cals = {}
    for i, er in enumerate(eratios):

        umatch = np.where( np.isclose(uratios, er, rtol=match_thresh) )
        e1, e2 = ecom[i][0], ecom[i][1]
        if test:
            print(f"\nratio {i} -- e1 {e1:.0f}  e2 {e2:.0f} -- {er:.3f}")

        if len(umatch[0]) == 0:
            continue

        caldists = []
        for ij, j in enumerate(umatch[0]):
            u1, u2 = ucom[j][0], ucom[j][1]
            cal = (e2 - e1) / (u2 - u1)
            cal_maxes = cal * umaxes

            # shift peaks by the amount we would expect if this const were true.
            # compute the distance (in "keV") of the peak that minimizes this.
            dist = 0
            for e_true in epeaks:
                idx = np.abs(cal_maxes - e_true).argmin()
                dist += np.abs(cal_maxes[idx] - e_true)
            caldists.append([cal, dist])

            if test:
                dev = er - uratios[j] # set by match_thresh parameter
                print(f"{ij}  {u1:-5.0f}  {u2:-5.0f}  {dev:-7.3f}  {cal:-5.2f}")

        # get the cal ratio with the smallest total dist
        caldists = np.array(caldists)
        imin = caldists[:,1].argmin()
        cals[i] = caldists[imin, :]

        if test:
            print(f"best: {imin}  {caldists[imin, 0]:.4f}  {caldists[imin, 1]:.4f}")

    if test:
        print("\nSummary:")
        for ipk in cals:
            e1, e2 = ecom[ipk][0], ecom[ipk][1]
            print(f"{ipk}  {e1:-6.1f}  {e2:-6.1f}  cal {cals[ipk][0]:.5f}")

    # get first-pass const for this DataSet
    cal_vals = np.array([c[1][0] for c in cals.items()])
    ds_cal = np.median(cal_vals)
    ds_std = np.std(cal_vals)
    print(f"Pass-1 cal for {etype}: {ds_cal:.5e} pm {ds_std:.5e}")

    if test:
        plt.semilogy(b * ds_cal, h, ls='steps', lw=1.5, c='b',
                     label=f"{etype}, {sum(h)} cts")
        for x,y in maxes:
            plt.plot(x * ds_cal, y, "m.", ms=10)

        pks = ds.config["pks"]
        cmap = plt.cm.get_cmap('jet', len(pks) + 1)
        for i, pk in enumerate(pks):
            plt.axvline(float(pk), c=cmap(i), linestyle="--", lw=1, label=f"{pks[pk]}: {pk} keV")

        plt.xlabel("Energy (keV)", ha='right', x=1)
        plt.ylabel("Counts", ha='right', y=1)
        plt.legend(fontsize=9)
 #       plt.show()

    if write_db:
        calDB = ds.calDB
        query = db.Query()
        table = calDB.table("cal_pass1")

        # write an entry for every dataset.  if we've chained together
        # multiple datasets, the values will be the same.
        # use "upsert" to avoid writing duplicate entries.
        for dset in ds.ds_list:
            row = {"ds":dset, "p1cal":ds_cal, "p1std":ds_std}
            table.upsert(row, query.ds == dset)


def calibrate_pass2(ds,df,run, cal_db,run_db,write_db=False):
    """
    load first-pass constants from the calDB for this DataSet,
    and the list of peaks we want to fit from the runDB, and
    fit the radford peak to each one.
    make a new table in the calDB, "cal_pass2" that holds all
    the important results, like mu, sigma, errors, etc.
    """
    print("Cal2")
    # take calibration parameter for the 'calibration.py' output
    with open(cal_db) as f:
      calDB = json.load(f)
    
    with open(run_db) as f:
      runDB = json.load(f)

    path_to_files =  os.path.expandvars(runDB["meta_dir"]) 
    true_peaks = sorted(ds.config["cal_peaks"], reverse=True)
    iter = 0
    
    plt.figure(1)
    peaks = []
    fwhms = []
    
    for true_peak in true_peaks:
      iter = iter+1
      ax = plt.subplot(3,2,iter)
      out1, out2 = peak(df,runDB,calDB,run,true_peak, plotit=True)
      peaks.append(out1)
      fwhms.append(out2)

    peaks = np.array(peaks,dtype=float)
    fwhms = np.array(fwhms,dtype=float)
    res = np.subtract(true_peaks,peaks)


    def pol1(x,a,b):
      return a * x + b

    pars1, cov1 = opt.curve_fit(pol1,peaks,true_peaks)
    errs1 = np.sqrt(np.diag(cov1))
    print("Calibration curve: ",pars1,errs1,cov1)

    peaks_1 = []
    fwhms_1 = []

    iter = 0
    for true_peak in true_peaks:
      iter = iter+1
      ax = plt.subplot(3,2,iter)
      out1, out2 = peak(df,runDB,calDB,run, true_peak,p=pars1, plotit =False)
      peaks_1.append(out1)
      fwhms_1.append(out2)


    plt.figure(2)
    plt.subplot(211)
    plt.plot(true_peaks,peaks,marker='o',linestyle='--',color='blue')
    plt.grid(True)
    plt.xlabel("True energy (keV)", ha='right', x=1)
    plt.ylabel("Energy (keV)", ha='right', y=1)
    plt.subplot(212)
    plt.plot(true_peaks,res,marker='o',linestyle='--',color='blue')
    plt.grid(True)
    plt.xlabel("True energy (keV)", ha='right', x=1)
    plt.ylabel("Residuals (keV)", ha='right', y=1)
    plt.savefig(path_to_files+'plots/energyScale_'+str(run)+'.png', bbox_inches='tight', transparent=True)

    def sqrt_fit(x,a,b):
      return np.sqrt(a*x+b)

    pars2, cov2 = opt.curve_fit(sqrt_fit,peaks_1,fwhms_1,p0=[1e-3,0.05])
    errs2 = np.sqrt(np.diag(cov2))
    print("Energy resolution curve: ",pars2,errs2)
    
    model = np.zeros(len(peaks))
    for i,bin in enumerate(peaks):
      model[i] = sqrt_fit(bin,pars2[0],pars2[1])
   

    print("Model", model) 
    plt.figure(3)
    plt.plot(sorted(peaks_1),sorted(fwhms_1),marker='o',linestyle='--',color='blue')
    #plt.savefig('./plots/energyScale.pdf', bbox_inches='tight', transparent=True)
    #plt.plot(sorted(peaks_1),sorted(model),'-',color='red')
    plt.grid(True)
    plt.xlabel("Energy (keV)", ha='right', x=1)
    plt.ylabel("FWHM resolution (keV) ", ha='right', y=1)
    plt.savefig(path_to_files+'plots/energyResolution_curve_'+str(run)+'.pdf', bbox_inches='tight', transparent=True)
    plt.show()

    if write_db:
      calDB = ds.calDB
      query = db.Query()
      table = calDB.table("cal_pass2")
        
      # write an entry for every dataset.  if we've chained together
      # multiple datasets, the values will be the same.
      # use "upsert" to avoid writing duplicate entries.
      for dset in ds.ds_list:
        row = {
          "ds":dset,
          "p2acal":pars1[0],
          "p2astd":errs1[0],
          "p2bcal":pars1[1],
          "p2bstd":errs1[1]
        }
        table.upsert(row, query.ds == dset)

        table = calDB.table("eres_curve")
        row = {
          "ds":dset,
          "acal":pars2[0],
          "astd":errs2[0],
          "bcal":pars2[1],
          "bstd":errs2[1]
        }
        table.upsert(row, query.ds == dset)

def peak(df,runDB,calDB, r,line,p=[1,0], plotit = False):

     
    cal = 0.04998# calDB["cal_pass1"]["1"]["p1cal"]
    meta_dir = os.path.expandvars(runDB["meta_dir"])
    tier_dir = os.path.expandvars(runDB["tier2_dir"])

    df['e_cal'] = p[0]*(cal*df['e_ftp']) + p[1]
    #h = df.hist('e_cal',bins=2000)
    #plt.yscale('log')
    df = df.loc[(df.index>1000)&(df.index<500000)]

    def gauss(x, mu, sigma, A=1):
      """
       define a gaussian distribution, w/ args: mu, sigma, area (optional).
       """
      return A * (1. / sigma / np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2. * sigma**2))
        
    line_min = 0.995*line
    line_max = 1.005*line
    nbin = 60
    res = 6.3e-4*line+0.85 # empirical energy resolution curve from experience
              
    hist, bins, var = pgh.get_hist(df['e_cal'], range=(line_min,line_max), dx=(line_max-line_min)/nbin)
    if plotit:
         pgh.plot_hist(hist, bins, var=hist, label="data", color='blue')
    pars, cov = pga.fit_hist(gauss, hist, bins, var=hist, guess=[line, res, 50])
    pgu.print_fit_results(pars, cov, gauss)
    if plotit:
         pgu.plot_func(gauss, pars, label="chi2 fit", color='red')
    
    FWHM = '%.2f' % Decimal(pars[1]*2.*np.sqrt(2.*np.log(2))) # convert sigma to FWHM
    FWHM_uncertainty = '%.2f' % Decimal(np.sqrt(cov[1][1])*2.*np.sqrt(2.*np.log(2)))
    peak = '%.2f' % Decimal(pars[0])
    peak_uncertainty = '%.2f' % Decimal(np.sqrt(cov[0][0]))
    residual = '%.2f' % abs(line - float(peak))
    
    if plotit:
        label_01 = 'Peak = '+str(peak)+r' $\pm$ '+str(peak_uncertainty)
        label_02 = 'FWHM = '+str(FWHM)+r' $\pm$ '+str(FWHM_uncertainty)
        labels = [label_01, label_02,]
    
        plt.xlim(line_min,line_max)
        plt.xlabel('Energy (keV)', ha='right', x=1.0)
        plt.ylabel('Counts', ha='right', y=1.0)
    
        plt.tight_layout()
        plt.hist(df['e_cal'],range=(line_min,line_max), bins=nbin)
        plt.legend(labels, frameon=False, loc='upper right', fontsize='small')

        plt.savefig(meta_dir+'/plots/lineFit_'+str(r)+'.png')    
    return peak, FWHM


def show_calDB(fdb):
    """
    pretty-print what we've stored in our calibration database
    """
    calDB = db.TinyDB(fdb)
    query = db.Query()
    table = calDB.table("cal_pass1")
    df = pd.DataFrame(table.all())
    print(df)

def show_calspectrum(ds,df, fdb, run, etype="e_ftp",p1=True,p2=False):
    """
    display the linearly calibrated energy spectrum
    """
      
    calDB = db.TinyDB(fdb)
    query = db.Query()
    table = calDB.table("cal_pass1")
    vals1 = table.all()
    if(p2):
      table = calDB.table("cal_pass2")
      vals2 = table.all()

    print("CalVals=",vals1)
    energy = df[etype]*vals1[0]['p1cal']
    if(p2):
      energy = energy/vals2[0]['p2acal'] - vals2[0]['p2bcal']
    hist, bins, var = pgh.get_hist(energy,range=[0,4000],dx=1)
    plt.plot(hist)
    plt.yscale('log')
    plt.savefig(path_to_files +'plots/calEnergy_spectrum_'+str(run)+'.png', bbox_inches='tight', transparent=True)
#    plt.show()

# need to display an estimate for the peakdet threshold
# based on the number of counts in each bin or something


if __name__=="__main__":
    main()
