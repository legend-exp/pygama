#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import tinydb as db
import matplotlib.pyplot as plt
import itertools as it
from scipy.stats import mode
from pprint import pprint
from pygama import DataSet
from pygama.utils import set_plot_style, peakdet
set_plot_style('clint')

def main():
    """
    perform automatic calibration of pygama DataSets.
    command line options to specify the DataSet are the same as in processing.py
    save results in a JSON database for access by other routines.
    """
    run_db, cal_db = "runDB.json", "calDB.json"

    par = argparse.ArgumentParser(description="calibration suite for MJ60")
    arg, st, sf = par.add_argument, "store_true", "store_false"
    arg("-ds", nargs='*', action="store", help="load runs for a DS")
    arg("-r", "--run", nargs=1, help="load a single run")
    arg("-s", "--spec", action=st, help="print simple spectrum")
    arg("-p1", "--pass1", action=st, help="run pass-1 (linear) calibration")
    arg("-p2", "--pass2", action=st, help="run pass-2 (peakfit) calibration")
    arg("-e", "--etype", nargs=1, help="custom energy param (default is e_ftp)")
    arg("-t", "--test", action=st, help="set verbose (testing) output")
    arg("-db", "--writeDB", action=st, help="store results in DB")
    arg("-pr", "--printDB", action=st, help="print calibration results in DB")
    args = vars(par.parse_args())

    # -- declare the DataSet --
    if args["ds"]:
        ds_lo = int(args["ds"][0])
        try:
            ds_hi = int(args["ds"][1])
        except:
            ds_hi = None
        ds = DataSet(ds_lo, ds_hi,
                     md=run_db, cal=cal_db, v=args["test"])

    if args["run"]:
        ds = DataSet(run=int(args["run"][0]), sub='none',
                     md=run_db, cal=cal_db, v=args["test"])

    # -- start calibration routines --
    etype = args["etype"][0] if args["etype"] else "e_ftp"

    if args["spec"]:
        show_spectrum(ds, etype)

    if args["pass1"]:
        calibrate_pass1(ds, etype, args["writeDB"], args["test"])

    if args["pass2"]:
        calibrate_pass2(ds, args["test"])

    # fit to germanium peakshape function goes here -- take from matthew's code
    # if args["pass3"]:
    #     calibrate_pass3(ds)

    if args["printDB"]:
        show_calDB(cal_db)



def show_spectrum(ds, etype="e_ftp"):
    """
    display the raw spectrum of an (uncalibrated) energy estimator,
    use it to tune the x-axis range and binning in preparation for
    the first pass calibration.
    """
    df = ds.get_t2df()
    print(df.columns)
    df.hist(etype)
    plt.show()

    # need to display an estimate for the peakdet threshold
    # based on the number of counts in each bin or something


def calibrate_pass1(ds, etype="e_ftp", write_db=False, test=False):
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
    maxes, mins = peakdet(h, pk_thresh, b)
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


def calibrate_pass2(ds, test=False):
    """
    load first-pass constants from the calDB for this DataSet,
    and the list of peaks we want to fit from the runDB, and
    fit the radford peak to each one.
    make a new table in the calDB, "cal_pass2" that holds all
    the important results, like mu, sigma, errors, etc.
    """

    """
    This function is mainly being used to estimate the FWHM of the calibration
    peaks
    """
    epeaks = sorted(ds.config["expected_peaks"])
    calpars = ds.get_p1cal_pars("e_ftp")
    xlo, xhi, xpb = calpars["xlims"]
    pk_thresh = calpars["width_thresh"]
    width_lo, width_hi, wlo1, whi1, wlo2, whi2 = calpars["width_lims"]

    calDB = ds.calDB
    query = db.Query()
    table = calDB.table("cal_pass1")
    vals = table.all()
    df_cal = pd.DataFrame(vals) # <<---- omg awesome
    df_cal = df_cal.loc[df_cal.ds.isin(ds.ds_list)]
    p1cal = df_cal.iloc[0]["p1cal"]

    t2 = ds.get_t2df()
    ene = t2["e_ftp"] * p1cal

    for i in range(len(epeaks)):
        ehi = epeaks[i] + width_hi
        elo = epeaks[i] + width_lo
        xpb = 1
        nb = int((ehi-elo)/xpb)
        h, bins = np.histogram(ene, nb, (elo, ehi))
        b = (bins[:-1] + bins[1:]) / 2

        # subract background
        mean_upper = np.mean(np.array(h[wlo1:whi1]))
        mean_lower = np.mean(np.array(h[wlo2:whi2]))
        h = h - ( mean_upper + mean_lower ) / 2

        max, min = peakdet(h, pk_thresh, b)
        print(max)

        binr = np.where(b == max[0][0])
        binl = np.where(b == max[0][0])
        binr, binl = binr[0], binl[0]
        peakh = max[0][1]
        fwhmr = h[binr]
        fwhml = h[binl]

        while fwhmr > 0.5 * peakh or fwhml > 0.5 * peakh:
            binr += 1
            binl += -1
            fwhmr = h[binr]
            fwhml = h[binl]

        print("FWHM is kind of in the ball park of: ", (b[binr]-b[binl]))

        if test:
            plt.plot(b, h, ls="steps", linewidth=1.5)
            plt.axvline(float(max[0][0]), c='red', linestyle="--", lw=1)
            plt.axvline(float(b[binr]), c='black', linestyle="--", lw=1)
            plt.axvline(float(b[binl]), c='green', linestyle="--", lw=1)
            plt.title("Peak: {}".format(epeaks[i]))
            plt.xlabel("keV")
            plt.ylabel("Counts/keV")
            plt.show()
    exit()

    # maxes, mins = peakdet(h, pk_thresh, b)



def show_calDB(fdb):
    """
    pretty-print what we've stored in our calibration database
    """
    calDB = db.TinyDB(fdb)
    query = db.Query()
    table = calDB.table("cal_pass1")
    df = pd.DataFrame(table.all())
    print(df)


if __name__=="__main__":
    main()
