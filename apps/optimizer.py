#!/usr/bin/env python3
import os
import argparse
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('clint.mpl')
from pprint import pprint
import scipy.signal as signal
import itertools

from pygama import DataSet
import pygama.utils as pu
import pygama.analysis.histograms as ph
import pygama.analysis.peak_fitting as pf


def main():
    """
    to get the best energy resolution, we want to explore the possible values
    of our DSP processor list, especially trap filter and RC decay constants.
    
    a flexible + easy way to vary a bunch of parameters at once is to create
    a DataFrame with each row corresponding to a set of parameters.  
    We then use this DF as an input/output for the other functions.
    
    it could also easily be extended to loop over individual detectors, or vary
    any other set of parameters in the processor list ......
    """
    par = argparse.ArgumentParser(description="pygama dsp optimizer")
    arg, st, sf = par.add_argument, "store_true", "store_false"
    arg("-ds", nargs='*', action="store", help="load runs for a DS")
    arg("-r", "--run", nargs=1, help="load a single run")
    arg("-g", "--grid", action=st, help="set DSP parameters to be varied")
    arg("-w", "--window", action=st, help="generate a small waveform file")
    arg("-p", "--process", action=st, help="run DSP processing")
    arg("-f", "--fit", action=st, help="fit outputs to peakshape function")
    arg("-t", "--plot", action=st, help="find optimal parameters & make plots")
    arg("-v", "--verbose", action=st, help="set verbose mode")
    args = vars(par.parse_args())
    
    ds = pu.get_dataset_from_cmdline(args, "runDB.json", "calDB.json")
    # pprint(ds.paths)
    
    # set I/O locations
    d_out = os.path.expanduser('~') + "/Data/cage"
    f_grid = f"{d_out}/cage_optimizer_grid.h5"
    f_tier1 = f"{d_out}/cage_optimizer_t1.h5"
    f_tier2 = f"{d_out}/cage_optimizer_t2.h5"
    f_opt = f"{d_out}/cage_optimizer_data.h5"

    # -- run routines --
    if args["grid"]:
        # set the combination of processor params to vary to optimize resolution
        set_grid(f_grid)
    
    if args["window"]:
        # generate a small single-peak file w/ uncalibrated energy to reanalyze
        window_ds(ds, f_tier1)
    
    if args["process"]:
        # create a file with DataFrames for each set of parameters
        process_ds(ds, f_grid, f_opt, f_tier1, f_tier2)
    
    if args["fit"]:
        # fit all outputs to the peakshape function and find the best resolution
        get_fwhm(f_grid, f_opt, verbose=args["verbose"])
    
    if args["plot"]:
        # show results
        plot_fwhm(f_grid) 
    
    
def set_grid(f_grid):
    """
    """
    # # this is pretty ambitious, but maybe doable -- 3500 entries
    # e_rises = np.arange(1, 6, 0.2)
    # e_flats = np.arange(0.5, 4, 0.5)
    # rc_consts = np.arange(50, 150, 5) # ~same as MJD charge trapping correction
    
    # this runs more quickly -- 100 entries, 3 minutes on my mac
    e_rises = np.arange(2, 3, 0.2)
    e_flats = np.arange(1, 3, 1)
    rc_consts = np.arange(52, 152, 10)
    
    # TODO: jason's suggestions, knowing the expected shape of the noise curve
    # e_rises = np.linspace(-1, 0, sqrt(sqrt(3)) # jason says try this
    # e_rises # make another list which is 10^pwr of this list
    # np.linspace(log_tau_min, log_tau_max) # jason says try this too
    
    lists = [e_rises, e_flats, rc_consts]
    
    prod = list(itertools.product(*lists)) # clint <3 stackoverflow
    
    df = pd.DataFrame(prod, columns=['rise','flat','rc']) 
    
    # print(df)

    df.to_hdf(f_grid, key="pygama_optimization")
    
    print("Wrote master grid file:", f_grid)

    
def window_ds(ds, f_tier1):
    """
    Take a single DataSet and window it so that the output file only contains 
    events near an expected peak location.
    """
    # a user has to figure out the uncalibrated energy range of the K40 peak
    # xlo, xhi, xpb = 0, 2e6, 2000 # show phys. spectrum (top feature is 2615 pk)
    xlo, xhi, xpb = 990000, 1030000, 250 # k40 peak, ds 3

    t2df = ds.get_t2df()
    hE, xE = ph.get_hist(t2df["energy"], range=(xlo, xhi), dx=xpb)
    plt.semilogy(xE, hE, ls='steps', lw=1, c='r')
    
    import matplotlib.ticker as ticker
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.4e'))
    plt.locator_params(axis='x', nbins=5)

    plt.xlabel("Energy (uncal.)", ha='right', x=1)
    plt.ylabel("Counts", ha='right', y=1)
    plt.savefig(f"./plots/cage_ds{ds.ds_lo}_winK40.pdf")
    # exit()
        
    # write a windowed tier 1 file containing only waveforms near the peak
    t1df = pd.DataFrame()
    for run in ds.paths:
        ft1 = ds.paths[run]["t1_path"]
        print(f"Scanning ds {ds.ds_lo}, run {run}\n    file: {ft1}")
        for chunk in pd.read_hdf(ft1, 'ORSIS3302DecoderForEnergy', chunksize=5e4):
            t1df_win = chunk.loc[(chunk.energy > xlo) & (chunk.energy < xhi)]
            print(t1df_win.shape)
            t1df = pd.concat([t1df, t1df_win], ignore_index=True)
    
    # -- save to HDF5 output file -- 
    h5_opts = {
        "mode":"w", # overwrite existing
        "append":False, 
        "format":"table",
        # "complib":"blosc:zlib", # no compression, increases I/O speed
        # "complevel":1,
        # "data_columns":["ievt"]
        }
    t1df.reset_index(inplace=True)
    t1df.to_hdf(f_tier1, key="df_windowed", **h5_opts)
    print("wrote file:", f_tier1)


def process_ds(ds, f_grid, f_opt, f_tier1, f_tier2):
    """
    and determine the trapezoid parameters that minimize
    the FWHM of the peak (fitting to the peakshape function).
    
    NOTE: I don't think we need to multiprocess this, since that's already
    being done in ProcessTier1
    """
    from pygama.dsp.base import Intercom
    from pygama.io.tier1 import ProcessTier1
    import pygama.io.decoders.digitizers as pgd
    
    df_grid = pd.read_hdf(f_grid)
    
    if os.path.exists(f_opt):
        os.remove(f_opt)
        
    # check the windowed file
    # tmp = pd.read_hdf(f_tier1)
    # nevt = len(tmp)

    t_start = time.time()
    
    for i, row in df_grid.iterrows():
        
        # estimate remaining time in scan
        if i == 4:
            diff = time.time() - t_start
            tot = diff/5 * len(df_grid) / 60
            tot -= diff / 60
            print(f"Estimated remaining time: {tot:.2f} mins")
        
        rise, flat, rc = row
        print(f"Row {i}/{len(df_grid)},  rise {rise}  flat {flat}  rc {rc}")
        
        # custom tier 1 processor list -- very minimal
        proc_list = {
            "clk" : 100e6,
            "fit_bl" : {"ihi":500, "order":1},
            "blsub" : {},
            "trap" : [
                {"wfout":"wf_etrap", "wfin":"wf_blsub", 
                 "rise":rise, "flat":flat, "decay":rc},
                {"wfout":"wf_atrap", "wfin":"wf_blsub", 
                 "rise":0.04, "flat":0.1, "fall":2} # could vary these too
                ],
            "get_max" : [{"wfin":"wf_etrap"}, {"wfin":"wf_atrap"}],
            # "ftp" : {"test":1}
            "ftp" : {}
        }
        proc = Intercom(proc_list)
        
        dig = pgd.SIS3302Decoder
        dig.decoder_name = "df_windowed"
        dig.class_name = None
        out_dir = "/".join(f_tier2.split("/")[:-1])
        
        # process silently
        ProcessTier1(f_tier1, proc, output_dir=out_dir, overwrite=True, 
                     verbose=False, multiprocess=True, nevt=np.inf, ioff=0, 
                     chunk=ds.config["chunksize"], run=ds.runs[0], 
                     t2_file=f_tier2, digitizers=[dig])
        
        # load the temporary file and append to the main output file
        df_key = f"opt_{i}"
        t2df = pd.read_hdf(f_tier2)
        t2df.to_hdf(f_opt, df_key)


def get_fwhm(f_grid, f_opt, verbose=False):
    """
    duplicate the plot from Figure 2.7 of Kris Vorren's thesis (and much more!)
    
    this code fits the e_ftp peak to the HPGe peakshape function (same as in
    calibration.py) and writes a new column to df_grid, "fwhm".
    """
    df_grid = pd.read_hdf(f_grid)

    # declare some new columns for df_grid
    cols = ["fwhm", "rchi2"]
    for col in cols:
        df_grid[col] = np.nan
    
    # loop over the keys and fit each e_ftp spectrum to the peakshape function
    print("i  rise  flat  rc  fwhm  rchi2")
    
    for i, row in df_grid.iterrows():
        
        key = f"opt_{i}"
        t2df = pd.read_hdf(f_opt, key=f"opt_{i}")
        
        # auto-histogram spectrum near the uncalibrated peak
        hE, xE, vE = ph.get_hist(t2df["e_ftp"], bins=1000, trim=False)
        
        # shift the histogram to be roughly centered at 0 and symmetric
        mu = xE[np.argmax(hE)]
        xE -= mu
        imax = np.argmax(hE)
        hmax = hE[imax]
        idx = np.where(hE > hmax/2) # fwhm
        ilo, ihi = idx[0][0], idx[0][-1]
        sig = (xE[ihi] - xE[ilo]) / 2.355
        idx = np.where((xE > -8 * sig) & (xE < 8 * sig))
        ilo, ihi = idx[0][0], idx[0][-1]-1

        xE = xE[ilo-1:ihi]
        hE, vE = hE[ilo:ihi], vE[ilo:ihi]

        # plt.plot(xE[1:], hE, ls='steps', c='r', lw=3)
        # plt.show()
        # exit()
        
        # set initial guesses for the peakshape function.  could all be improved
        mu = 0
        sigma = 5 # radford uses an input linear function
        hstep = 0.001 
        htail = 0.5
        tau = 10
        bg0 = np.mean(hE[:20])
        amp = np.sum(hE)
        x0 = [mu, sigma, hstep, htail, tau, bg0, amp]
        
        xF, xF_cov = pf.fit_hist(pf.radford_peak, hE, xE, var=vE, guess=x0)
        
        # goodness of fit
        chisq = []
        for j, h in enumerate(hE):
            model = pf.radford_peak(xE[j], *xF)
            diff = (model - h)**2 / model
            chisq.append(abs(diff))

        # update the master dataframe
        fwhm = xF[1] * 2.355
        rchi2 = sum(np.array(chisq) / len(hE))
        
        df_grid.at[i, "fwhm"] = fwhm
        df_grid.at[i, "rchi2"] = rchi2
        
        rise, flat, rc = row[:3]
        label = f"{i} {rise:.2f} {flat:.2f} {rc:.0f} {fwhm:.2f} {rchi2:.2f}"
        print(label)

        if verbose:

            # plot every dang fit 
            plt.cla()

            # peakshape function
            plt.plot(xE, pf.radford_peak(xE, *x0), c='orange', label='guess')
            plt.plot(xE, pf.radford_peak(xE, *xF), c='r', label='peakshape')
            
            plt.axvline(mu, c='g')
            
            # plot individual components
            # tail_hi, gaus, bg, step, tail_lo = pf.radford_peak(xE, *xF, components=True)
            # gaus = np.array(gaus)
            # step = np.array(step)
            # tail_lo = np.array(tail_lo)
            # plt.plot(xE, gaus * tail_hi, ls="--", lw=2, c='g', label="gaus+hi_tail")
            # plt.plot(xE, step + bg, ls='--', lw=2, c='m', label='step + bg')
            # plt.plot(xE, tail_lo, ls='--', lw=2, c='k', label='tail_lo')
        
            plt.plot(xE[1:], hE, ls='steps', lw=1, c='b', label="data")
            plt.plot(np.nan, np.nan, c='w', label=f"fwhm = {fwhm:.2f} uncal.")
            plt.plot(np.nan, np.nan, c='w', label=label)
        
            plt.xlabel("Energy (uncal.)", ha='right', x=1)
            plt.ylabel("Counts", ha='right', y=1)
            plt.legend(loc=2, fontsize=12)
            
            plt.show()
            
    # write the updated df_grid to the output file.  
    if not verbose:
        df_grid.to_hdf(f_grid, key="pygama_optimization")
        print("wrote output file")
    
            
def plot_fwhm(f_grid):
    """
    """
    df_grid = pd.read_hdf(f_grid)

    # find overall minimum values
    df_min = df_grid.loc[df_grid['fwhm'].idxmin()]
    rise, flat, rc = df_min[:3]
    
    # 1. vary the rise time
    df_rise = df_grid.loc[(df_grid.flat==flat)&(df_grid.rc==rc)]
    plt.plot(df_rise.rise, df_rise.fwhm**2, ".b")
    # plt.plot(df_rise.rise, df_rise.fwhm, ".b")
    plt.xlabel("Ramp time (us)", ha='right', x=1)
    plt.ylabel(r"FWHM$^2$ (uncal)", ha='right', y=1)
    # plt.ylabel(r"FWHM", ha='right', y=1)
    plt.savefig("./plots/cage_optimizer_rise.pdf")
    plt.cla()
    
    # 2. vary the flat time
    df_flat = df_grid.loc[(df_grid.rise==rise)&(df_grid.rc==rc)]
    plt.plot(df_flat.flat, df_flat.fwhm, ".b")
    plt.xlabel("Flat time (us)", ha='right', x=1)
    plt.ylabel("FWHM (uncal)", ha='right', y=1)
    plt.savefig("./plots/cage_optimizer_flat.pdf")
    plt.cla() 
    
    # 3. vary the rc constant
    df_rc = df_grid.loc[(df_grid.rise==rise)&(df_grid.flat==flat)]
    plt.plot(df_rc.rc, df_rc.fwhm, ".b")
    plt.xlabel("RC constant (us)", ha='right', x=1)
    plt.ylabel(r"FWHM (uncal)", ha='right', y=1)
    plt.savefig("./plots/cage_optimizer_rc.pdf")
    plt.cla()
    
    
if __name__=="__main__":
    main()