#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('clint.mpl')
from pprint import pprint
import scipy.signal as signal

from pygama import DataSet
import pygama.utils as pu
import pygama.analysis.histograms as ph
import pygama.analysis.peak_fitting as pf


def main():
    """
    NOTE: We could also optimize the A trap here, it might help with A/E
    """
    # window_ds()
    
    # values to loop over -- might want to zip them together into tuples
    rise_times = [1, 2, 3, 4, 5]
    
    process_ds(rise_times)
    optimize_trap(rise_times, True)
    
    
def window_ds():
    """
    Take a single DataSet and window it so that the file only contains events 
    near an expected peak location.
    Create some temporary in/out files s/t the originals aren't overwritten.
    """
    # run = 42
    # ds = DataSet(run=run, md="runDB.json")
    ds_num = 3
    ds = DataSet(ds_num, md="runDB.json")
    
    # specify temporary I/O locations
    p_tmp = "~/Data/cage"
    f_tier1 = "~/Data/cage/cage_ds3_t1.h5"
    f_tier2 = "~/Data/cage/cage_ds3_t2.h5"
    
    # figure out the uncalibrated energy range of the K40 peak
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
    plt.savefig(f"./plots/cage_ds{ds_num}_winK40.pdf")
    # exit()
        
    # write a windowed tier 1 file containing only waveforms near the peak
    t1df = pd.DataFrame()
    for run in ds.paths:
        ft1 = ds.paths[run]["t1_path"]
        print(f"Scanning ds {ds_num}, run {run}\n    file: {ft1}")
        for chunk in pd.read_hdf(ft1, 'ORSIS3302DecoderForEnergy', chunksize=5e4):
            t1df_win = chunk.loc[(chunk.energy > xlo) & (chunk.energy < xhi)]
            print(t1df_win.shape)
            t1df = pd.concat([t1df, t1df_win], ignore_index=True)
    
    # -- save to HDF5 output file -- 
    h5_opts = {
        "mode":"w", # overwrite existing
        "append":False, 
        "format":"table",
        "complib":"blosc:zlib",
        "complevel":1,
        "data_columns":["ievt"]
        }
    t1df.reset_index(inplace=True)
    t1df.to_hdf(f_tier1, key="df_windowed", **h5_opts)
    print("wrote file:", f_tier1)


def process_ds(rise_times):
    """
    and determine the trapezoid parameters that minimize
    the FWHM of the peak (fitting to the peakshape function).
    """
    from pygama.dsp.base import Intercom
    from pygama.io.tier1 import ProcessTier1
    import pygama.io.decoders.digitizers as pgd
    
    ds_num = 3
    ds = DataSet(ds_num, md="runDB.json")
    first_run = ds.runs[0]
    
    # specify temporary I/O locations
    out_dir = os.path.expanduser('~') + "/Data/cage"
    t1_file = f"{out_dir}/cage_ds3_t1.h5"
    t2_file = f"{out_dir}/cage_ds3_t2.h5"
    opt_file = f"{out_dir}/cage_ds3_optimize.h5"
    
    if os.path.exists(opt_file):
        os.remove(opt_file)
        
    # check the windowed file
    tmp = pd.read_hdf(t1_file)
    nevt = len(tmp)

    rc_decay = 72
    
    for i, rt in enumerate(rise_times):
        
        # custom tier 1 processor list -- very minimal
        proc_list = {
            "clk" : 100e6,
            "fit_bl" : {"ihi":500, "order":1},
            "blsub" : {},
            "trap" : [
                {"wfout":"wf_etrap", "wfin":"wf_blsub", 
                 "rise":rt, "flat":2.5, "decay":rc_decay},
                {"wfout":"wf_atrap", "wfin":"wf_blsub", 
                 "rise":0.04, "flat":0.1, "fall":2}
                ],
            "get_max" : [{"wfin":"wf_etrap"}, {"wfin":"wf_atrap"}],
            # "ftp" : {"test":1}
            "ftp" : {}
        }
        proc = Intercom(proc_list)
        
        dig = pgd.SIS3302Decoder
        dig.decoder_name = "df_windowed"
        dig.class_name = None
        
        ProcessTier1(t1_file, proc, output_dir=out_dir, overwrite=True, 
                     verbose=False, multiprocess=True, nevt=np.inf, ioff=0, 
                     chunk=ds.config["chunksize"], run=first_run, 
                     t2_file=t2_file, digitizers=[dig])
        
        # load the temporary file and append to the main output file
        df_key = f"opt_{i}"
        t2df = pd.read_hdf(t2_file)
        t2df.to_hdf(opt_file, df_key)


def optimize_trap(rise_times, test=False):
    """
    duplicate the plot from Figure 2.7 of Kris Vorren's thesis.
    need to fit the e_ftp peak to the HPGe peakshape function (same as in
    calibration.py) and plot the resulting FWHM^2 vs. the ramp time.
    """
    out_dir = "~/Data/cage"
    opt_file = f"{out_dir}/cage_ds3_optimize.h5"
    print("input file:", opt_file)
    
    # match keys to settings; should maybe do this in prev function as attrs.
    with pd.HDFStore(opt_file, 'r') as store:
        keys = [key[1:] for key in store.keys()]  # remove leading '/'
        settings = {keys[i] : rise_times[i] for i in range(len(keys))}
    
    # loop over the keys and fit each e_ftp spectrum to the peakshape function
    fwhms = {}
    for key, rt in settings.items():
        
        t2df = pd.read_hdf(opt_file, key=key)
        
        # histogram spectrum near the uncalibrated peak -- have to be careful here
        xlo, xhi, xpb = 2550, 2660, 1
        hE, xE, vE = ph.get_hist(t2df["e_ftp"], range=(xlo, xhi), dx=xpb, trim=False)
        
        # set initial guesses for the peakshape function.  most are pretty rough
        mu = xE[np.argmax(hE)]
        sigma = 5
        hstep = 0.001
        htail = 0.5
        tau = 10
        bg0 = np.mean(hE[:20])
        amp = np.sum(hE)
        x0 = [mu, sigma, hstep, htail, tau, bg0, amp]
        
        xF, xF_cov = pf.fit_hist(pf.radford_peak, hE, xE, var=vE, guess=x0)
        
        fwhms[key] = xF[1] * 2.355

        if test:
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
            plt.plot(np.nan, np.nan, c='w', label=f"fwhm = {results['fwhm']:.2f} uncal.")
        
            plt.xlabel("Energy (uncal.)", ha='right', x=1)
            plt.ylabel("Counts", ha='right', y=1)
            plt.legend(loc=2)
            
            plt.show()
            

    
if __name__=="__main__":
    main()