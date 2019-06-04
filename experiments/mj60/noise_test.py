#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools as it
from pprint import pprint
from scipy.signal import medfilt
from pygama import DataSet
from pygama.utils import set_plot_style, peakdet
from pygama.analysis.histograms import get_hist
set_plot_style('clint')

def main():
    """
    try a few noise rejection methods for mj60.
    i don't have access to the t1 dataframe at the moment, so use t2 only.
    data columns:
    ['channel', 'energy', 'energy_first', 'ievt', 'packet_id', 'timestamp',
       'ts_hi', 'bl_rms', 'bl_p0', 'bl_p1', 'etrap_max', 'etrap_imax',
       'strap_max', 'strap_imax', 'atrap_max', 'atrap_imax', 'ttrap_max',
       'ttrap_imax', 'savgol_max', 'savgol_imax', 'current_max',
       'current_imax', 'tp5', 'tp10', 'tp50', 'tp100', 'n_curr_pks',
       's_curr_pks', 't0', 't_ftp', 'e_ftp', 'overflow', 'tslope_savgol',
       'tslope_pz', 'tail_amp', 'tail_tau']
    """
    ds = DataSet(run=305, md='runDB.json')
    df = ds.get_t2df()

    calibrate(df)
    

def calibrate(df):
    """
    run a simple linear calibration of e_ftp.
    TODO: make constants on top input parameters (more flexible for all runs)
    """
    ene = df["e_ftp"]
    ecal = [2614.511, 1460.820] # gamma lines we assume are always present
    pk_thresh = 50
    match_thresh = 0.01
    xlo, xhi, xpb = 0, 10000, 10

    # make energy histogram
    nb = int((xhi-xlo)/xpb)
    h, bins = np.histogram(ene, nb, (xlo, xhi))
    b = (bins[:-1] + bins[1:]) / 2.
    
    # filter out continuum
    # hmed = medfilt(h, 11)
    # hpks = h - hmed
    
    # run peakdet to identify the maxima
    maxes, mins = peakdet(h, pk_thresh, b)
    xmaxes = [x[0] for x in maxes]
    
    # ratios of known gamma energies
    ecom = [c for c in it.combinations(ecal, 2)]
    erat = np.array([x[0] / x[1] for x in ecom])

    # ratios of peaks in the uncalibrated spectrum
    dcom = [c for c in it.combinations(xmaxes, 2)]
    drat = np.array([x[1] / x[0] for x in dcom])

    # match peaks to known energies 
    for i, er in enumerate(erat):
        dmatch = np.where( np.isclose(drat, er, rtol=match_thresh) )
        if len(dmatch[0]) == 0:
            continue
        
        e1, e2 = ecom[i][0], ecom[i][1]
        print("e1 {:.0f}  e2 {:.0f}  erat {:.3f}  idxs:".format(e1, e2, er), dmatch)
        
        for didx in dmatch[0]:
            d1, d2 = dcom[didx][1], dcom[didx][0]
            print("d1 {:.0f}  d2 {:.0f}  drat {:.3f}".format(d1, d2, drat[didx]))

        print("")


    
    # plot uncalibrated maxima
    for x,y in maxes:
        plt.plot(x, y, "m.", ms=10)
        
    # plot spectrum
    plt.semilogy(b, h, ls='steps', lw=1.5, c='b', 
                 label="pygama e_ftp, {} cts".format(sum(h)))

        
    plt.xlabel("Energy (uncal)", ha='right', x=1)
    plt.ylabel("Counts", ha='right', y=1)
    plt.legend()
    plt.show()
    
    
    
    
if __name__=="__main__":
    main()
