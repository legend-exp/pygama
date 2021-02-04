#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import tinydb as db
import matplotlib.pyplot as plt
plt.style.use('clint.mpl')
import itertools as it
from scipy.stats import mode
from pprint import pprint

from pygama import DataSet
import pygama.utils as pu
import pygama.analysis.histograms as ph
import pygama.analysis.peak_fitting as pf

def main():
    """
    perform automatic calibration of pygama DataSets.
    command line options to specify the DataSet are the same as in processing.py
    save results in a JSON database for access by other routines.
    """
    run_db, cal_db = "runDB.json", "calDB.json"

    par = argparse.ArgumentParser(description="pygama calibration suite")
    arg, st, sf = par.add_argument, "store_true", "store_false"
    arg("-ds", nargs='*', action="store", help="load runs for a DS")
    arg("-r", "--run", nargs=1, help="load a single run")
    arg("-s", "--spec", action=st, help="print simple spectrum")
    arg("-p1", "--pass1", action=st, help="run pass-1 (linear) calibration")
    arg("-p2", "--pass2", action=st, help="run pass-2 (peakfit) calibration")
    arg("-m", "--mode", nargs=1, help="set pass-2 calibration mode")
    arg("-e", "--etype", nargs=1, help="custom energy param (default is e_ftp)")
    arg("-t", "--test", action=st, help="set verbose (testing) output")
    arg("-db", "--writeDB", action=st, help="store results in DB")
    arg("-pr", "--printDB", action=st, help="print calibration results in DB")
    args = vars(par.parse_args())

    # -- standard method to declare the DataSet from cmd line --
    ds = pu.get_dataset_from_cmdline(args, "runDB.json", "calDB.json")
    
    # -- start calibration routines --
    etype = args["etype"][0] if args["etype"] else "e_ftp"

    if args["printDB"]:
        show_calDB(cal_db) # print current DB status

    if args["spec"]:
        show_spectrum(ds, etype) 

    if args["pass1"]:
        calibrate_pass1(ds, etype, args["writeDB"], args["test"])

    if args["pass2"]:
        cal_mode = int(args["mode"][0]) if args["mode"] else 0
        calibrate_pass2(ds, cal_mode, args["writeDB"])


def show_calDB(fdb):
    """
    pretty-print what we've stored in our calibration database
    """
    calDB = db.TinyDB(fdb)
    query = db.Query()
    # table = calDB.table("cal_pass1")
    table = calDB.table("cal_pass2")
    df = pd.DataFrame(table.all())
    print(df)


def show_spectrum(ds, etype="e_ftp"):
    """
    display the raw spectrum of an (uncalibrated) energy estimator,
    use it to tune the x-axis range and binning in preparation for
    the first pass calibration.  
    
    TODO -- it would be neat to use this function to display an estimate for 
    the peakdet threshold we need later in the code, based on the number of 
    counts in each bin or something ...
    """
    t2df = ds.get_t2df()
    print(t2df.columns)
    ene = "e_ftp"
    
    # built-in pandas histogram
    # t2df.hist(etype, bins=1000)
    
    # pygama histogram
    xlo, xhi, xpb = 0, 6000, 10 # gamma spectrum
    hE, xE = ph.get_hist(t2df[ene], range=(xlo, xhi), dx=xpb)
    plt.semilogy(xE, hE, ls='steps', lw=1, c='r')

    plt.xlabel("Energy (uncal.)", ha='right', x=1)
    plt.ylabel("Counts", ha='right', y=1)
    plt.show()
    

def calibrate_pass1(ds, etype="e_ftp", write_db=False, test=False):
    """
    Run a "first guess" calibration of an arbitrary energy estimator.

    Uses a peak matching algorithm based on finding ratios of uncalibrated (u)
    and "true, keV-scale" (e) energies.  
    We run peakdet to find the maxima in the spectrum, then compute all ratios:
        - e1/e2, u1/u2, ..., u29/u30 etc.
    We find the subset of uncalibrated ratios (u7/u8, ... etc) that match the 
    "true" ratios, and compute a calibration constant for each.
    
    Then for each uncalibrated ratio, we assume it to be true, then loop over 
    the expected peak positions.
    
    We shift the uncalibrated peaks so that the true peak would be very close
    to 0, and calculate its distance from 0.  The "true" calibration constant
    will minimize this value for all ratios, and this is the one we select.

    Writes first-pass calibration results to a database, for access
    by the second pass, and other codes.
    """
    # get a list of peaks we assume are always present
    epeaks = sorted(ds.config["expected_peaks"], reverse=True)

    # get initial parameters for this energy estimator
    calpars = ds.get_p1cal_pars(etype)
    pk_thresh = calpars["peakdet_thresh"]
    match_thresh = calpars["match_thresh"]
    xlo, xhi, xpb = calpars["xlims"]

    # make energy histogram
    df = ds.get_t2df()
    ene = df[etype]
    nb = int((xhi-xlo)/xpb)
    h, bins = np.histogram(ene, nb, (xlo, xhi))
    b = (bins[:-1] + bins[1:]) / 2.

    # run peakdet to identify the uncalibrated maxima
    maxes, mins = pu.peakdet(h, pk_thresh, b)
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
        plt.show()

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


def calibrate_pass2(ds, mode, write_db=False):
    """
    Load first-pass constants from the calDB for this DataSet, and the list of 
    peaks we want to fit from the runDB, and fit the PPC peakshape to each one.
    
    Apply pygama fit functions developed in pygama.analysis.peak_fitting
    
    TODO: 
    Make a new table in the calDB for each DataSet, "cal_pass2", that 
    holds fit results, etc.  These should be used as inputs for the 
    MultiPeakFitter calibration code (pass 3).
    """
    etype, ecal = "e_ftp", "e_cal"
    
    # load calibration database file with tinyDB and convert to pandas
    calDB = ds.calDB
    query = db.Query()
    table = calDB.table("cal_pass1").all()
    df_cal = pd.DataFrame(table) # <<---- omg awesome
    
    # apply calibration from db to tier 2 dataframe
    df_cal = df_cal.loc[df_cal.ds.isin(ds.ds_list)]
    p1cal = df_cal.iloc[0]["p1cal"]
    t2df = ds.get_t2df()
    t2df[ecal] = t2df[etype] * p1cal # create a new column
    
    # get additional options from the config file
    cal_opts = ds.get_p1cal_pars(etype)
    pk_lim = cal_opts["peak_lim_keV"]
    # pk_thresh = cal_opts["peakdet_thresh"]

    fits = {}
    pk_names = ds.config["pks"]

    # loop over a list of peaks we assume are always present
    for e_peak in sorted(ds.config["main_peaks"], reverse=True):

        # histogram the spectrum near the peak
        xlo, xhi, xpb = e_peak - pk_lim, e_peak + pk_lim, 1
        hE, xE, vE = ph.get_hist(t2df[ecal], range=(xlo, xhi), dx=xpb, trim=False)
        
        # run peakdet and measure the difference between expected & calibrated
        # maxes, mins = pu.peakdet(hE, pk_thresh, xE)
        # diffs = [e_peak - pk_val[0] for pk_val in maxes]
        # pk_min, i_min = min((v, i) for (i, v) in enumerate(diffs))
        # print(e_peak, pk_min, i_min)
        
        # -- run gaussian fit (gauss + linear bkg term) -- 
        if mode == 0:
            
            # mu, sigma, a, b, m
            # TODO: could set initial sigma w. some simple linear function
            x0 = [e_peak, 5, np.sum(hE), np.mean(hE[:50]), 1]
            
            xF, xF_cov = pf.fit_hist(pf.gauss_lin, hE, xE, var=np.ones(len(hE)), 
                                     guess=x0)
            results = {
                "e_fit" : xF[0],
                "e_unc" : np.sqrt(xF_cov[0][0]),
                "fwhm" : xF[1] * 2.355,
                "fwhm_unc" : np.sqrt(xF_cov[1][1]) * 2.355,
                "resid" : abs(e_peak - xF[0]),
                "bkg0" : xF[3],
                "bkg1" : xF[4]
                }
            chisq = []
            for i, h in enumerate(hE):
                diff = (pf.gauss_lin(xE[i], *xF) - hE[i])**2 / pf.gauss_lin(xE[i], *xF)
                chisq.append(abs(diff))
            results["chisq_ndf"] = sum(np.array(chisq) / len(hE))
            
            # update DB results
            fits[pk_names[str(e_peak)]] = results
        
        # -- run peakshape function fit (+ linear bkg term) -- 
        elif mode == 1:
            
            # peakshape parameters: mu, sigma, hstep, htail, tau, bg0, a=1
            hstep = 0.001 # fraction that the step contributes
            htail = 0.1
            amp = np.sum(hE)
            tau = 10
            bg0 = np.mean(hE[:20])
            x0 = [e_peak, 5, hstep, htail, tau, bg0, amp]
            
            xF, xF_cov = pf.fit_hist(pf.radford_peak, hE, xE, var=vE, guess=x0)
            
            results = {
                "e_fit" : xF[0],
                "fwhm" : xF[1] * 2.355
                # ...
            }
            chisq = []
            for i, h in enumerate(hE):
                diff = (pf.radford_peak(xE[i], *xF) - hE[i])**2 / pf.radford_peak(xE[i], *xF)
                chisq.append(abs(diff))
            results["chisq_ndf"] = sum(np.array(chisq) / len(hE))
            
            # update DB results
            fits[pk_names[str(e_peak)]] = results


        # -- plot the fit -- 
        plt.axvline(e_peak, c='g')

        if mode==0:
            # gaussian fit
            # plt.plot(xE, pf.gauss_lin(xE, *x0), c='orange', label='guess')
            plt.plot(xE, pf.gauss_lin(xE, *xF), c='r', label='fit')
        
        if mode==1:
            # peakshape function
            # plt.plot(xE, pf.radford_peak(xE, *x0), c='orange', label='guess')
            plt.plot(xE, pf.radford_peak(xE, *xF), c='r', label='peakshape')
            
            # plot individual components
            
            # consts - tail_hi & bg
            tail_hi, gaus, bg, step, tail_lo = pf.radford_peak(xE, *xF, components=True)
            gaus = np.array(gaus)
            step = np.array(step)
            tail_lo = np.array(tail_lo)
            
            plt.plot(xE, gaus * tail_hi, ls="--", lw=2, c='g', label="gaus+hi_tail")
            plt.plot(xE, step + bg, ls='--', lw=2, c='m', label='step + bg')
            plt.plot(xE, tail_lo, ls='--', lw=2, c='k', label='tail_lo')
        
        plt.plot(xE[1:], hE, ls='steps', lw=1, c='b', label="data")
        plt.plot(np.nan, np.nan, c='w', label=f"fwhm = {results['fwhm']:.2f} keV")
        
        plt.xlabel("Energy (keV)", ha='right', x=1)
        plt.ylabel("Counts", ha='right', y=1)
        plt.legend()
        # plt.show()
        plt.savefig("./plots/cage_ds3_pass2cal.pdf")
        
        
    if write_db:
        calDB = ds.calDB
        query = db.Query()
        table = calDB.table("cal_pass2")
        
        # collapse data to 1 row
        row = {}
        for pk in fits:
            for key, val in fits[pk].items():
                row[f"{key}_{pk}"] = val
                
        # write an entry for every dataset.  if we've chained together
        # multiple datasets, the values will be the same.
        # use "upsert" to avoid writing duplicate entries.
        for dset in ds.ds_list:
            table.upsert(row, query.ds == dset)

        print("wrote results to DB.")
        

def calibrate_pass3(ds, df,etype="e_ftp", write_db=False, display=False, linfit = True):

    """
    This is the calibration method I used for HADES ICPC characterization  
    
    You have to look at the raw spectrum for one dataset once. One dataset implies:
 
     - One detector setup
     - One source
     - One daq-setting 

    From the raw spectrum at least two raw lines of your choice (i.e. 208Tl, 40K, Bi, etc) and determin an range 
    around the lines have to be selected and added to a config file (to the runDB.json in the HADES work)
    under "pass3_peaks" and "pass3_lim".
    (Note that you also need the literature values in e.g. "cal_peaks" ) 

    This calibration function will get the raw peaks by constructing a histogram around the raw values 
    ,perfom a fit and devide the literature value by the output (c = lit/out). 
    Then plot all received cal values vs energy and do a poly1 fit to extract the energy dependence (a*x+b)
    The fit function can also be of other type (quadratic, sqrt-like or else if needed, default = linfit)

    The calibrated energy is then ecal = eraw * (eraw *a +b)  
  

    A.Zschocke    
    """

    etype = "e_ftp"

    # get the list of peaks we want  
    epeaks = np.array(sorted(ds.runDB["pass3_peaks"]))
    e_lim = ds.runDB["pass3_lim"]
    true_peaks = sorted(np.array(ds.runDB["cal_peaks"]))

    # get the raw energy 
    ene = df[etype]
    means = []
    cals = []
    # do the firts loop over all lines of interest
    for i, peak in enumerate(epeaks):

        xlo, xhi, xpb = peak - e_lim, peak + e_lim, 0.25
        nb = int((xhi-xlo)/xpb)
        hE, xE, vE = ph.get_hist(ene, range=(xlo, xhi), dx=xpb)
        guess = [100000,peak,0.7,1000]
        guess_lims = ([0,peak -20,0,0],[1e9,peak+20,1e9,1e9])

        xF, xF_cov = pf.fit_hist(pf.gauss_bkg, hE, xE, var=np.ones(len(hE)), guess=guess, bounds=guess_lims)
        fit = pf.gauss_bkg(xE,*xF)

        if display:
           plt.plot(xE[1:],hE,ls='steps',lw='1.5',c='b')
           plt.plot(xE,fit,'r')
           plt.show()

        mean = xF[1]
        means.append(mean)
        cals.append(true_peaks[i]/mean)

    # now calculate the energy dependence
    cals = np.array(cals)
    means = np.array(means)

    if linfit:
       xF = np.polyfit(means,cals,1)
       pfit = means *xF[0] + xF[1]
       cal_peaks = means*(means*xF[0]+xF[1])

    else:
       xF, xF_coev = curve_fit(pf.cal_slope, epeaks, cals)
       pfit = pf.cal_slope(epeaks,*xF)
       cal_peaks = means*(np.sqrt(xF[0]+(xF[1]/means**2)))

    print(f"Calibration values:\n a={pfit[0]:.5f} b={pfit[1]:.5f}")

    residuals = abs((cal_peaks-true_peaks))#/true_peaks*100

    if any(residuals > 1):
       r = residuals[np.where(residuals > 0)]
       print("\nWaning! No proper calibration\nThere is a deviation of",r, "%")


    if display:

       meta_dir = os.path.expandvars(ds.runDB["meta_dir"])
       runNum = ds.ds_list[0]
       x = np.arange(1,60000,1)

       if linfit:
          e_cal = ene * (ene * xF[0] + xF[1])
          pfit = x * xF[0] + xF[1]
       else:
          e_cal = ene *np.sqrt(xF[0] +(xF[1]/(ene**2)))
          pfit = pf.cal_slope(x,*xF)

       hE, xE, vE = ph.get_hist(e_cal, range=(0, 3000), dx=1)

        x = np.arange(1,60000,1)

       plt.plot(epeaks, cals, 'kx', ms=10, label='calibration values')
       plt.plot(x, pfit,'r',label='Fit')
       plt.xlim(0,60010)
       plt.xlabel("raw Energy")
       plt.ylabel("cal. value")
       plt.legend()
       plt.savefig(meta_dir+"/calVals_pass3_" + str(runNum)+".png")
       plt.show()

       hE, xE, vE = ph.get_hist(e_cal, range=(0, 3000), dx=1)
       plt.figure(1,(12.00,10.00))
       plt.subplot(211)
       plt.semilogy(xE[1:], hE, ls='steps', lw=1.5,c='b')
       plt.xlabel("Energy [keV]")
       plt.xlim(-10,3100)
       plt.subplot(212)
       plt.plot(true_peaks, residuals, 'kx',ms=10,label='Residuals')
       plt.grid()
       plt.xlabel("Energy [keV]")
       plt.ylabel("Residuals [%]")
       plt.xlim(-10,3100)
       plt.legend()
       plt.savefig(meta_dir+"/calibratedSpectrum_pass3_" + str(runNum)+".png")
       plt.show()


    if write_db:
        calDB = ds.calDB
        query = db.Query()
        table = calDB.table("cal_pass3")

        for dset in ds.ds_list:
            row = {"ds":dset, "lin":linfit, "slope":xF[0], "offset":xF[1]}
            table.upsert(row, query.ds == dset)

if __name__=="__main__":
    main()
