import pandas as pd
import sys
import numpy as np
import scipy as sp
import json
import os
from decimal import Decimal
import scipy.optimize as opt
from scipy.optimize import minimize, curve_fit
from scipy.special import erfc
from scipy.stats import crystalball
from scipy.signal import medfilt, find_peaks
import pygama.analysis.histograms as pgh
import pygama.utils as pgu
import pygama.analysis.peak_fitting as pga
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():

    ## this code takes the peaks from thorium's first-pass calibration and fits them.

    lines = [238.4,583.191,860.564,2103.533,2614.533]
    iter = 0
    
    plt.figure(1)
    peaks = []
    FWHMs = []
    
    for line in lines:
      iter = iter+1
      ax = plt.subplot(3,2,iter)
      out1, out2 = peak(line)
      peaks.append(out1)
      FWHMs.append(out2)
    
    plt.figure(2)
    plt.plot(peaks,FWHMs,marker='o',linestyle='--',color='blue')
    plt.grid(True)
    plt.show()

def peak(line):
    if(len(sys.argv) != 2):
        print('Usage: fit_calibrated_peaks.py [run number]')
        sys.exit()

    # take calibration parameter for the 'calibration.py' output
    with open("calDB.json") as f:
      calDB = json.load(f)
    cal = calDB["cal_pass1"]["1"]["p1cal"]

    with open("runDB.json") as f:
        runDB = json.load(f)
    meta_dir = os.path.expandvars(runDB["meta_dir"])
    tier_dir = os.path.expandvars(runDB["tier_dir"])

    #df =  pd.read_hdf("{}/Spectrum_{}_2.hdf5".format(meta_dir,sys.argv[1]), key="df")
    df =  pd.read_hdf("{}/t2_run{}.h5".format(tier_dir,sys.argv[1]))

    df['e_cal'] = cal*df['e_ftp']

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
    pgh.plot_hist(hist, bins, var=hist, label="data", color='blue')
    pars, cov = pga.fit_hist(gauss, hist, bins, var=hist, guess=[line, res, 50])
    pgu.print_fit_results(pars, cov, gauss)
    pgu.plot_func(gauss, pars, label="chi2 fit", color='red')

    FWHM = '%.2f' % Decimal(pars[1]*2)
    FWHM_uncertainty = '%.2f' % Decimal(np.sqrt(cov[1][1])*2)
    peak = '%.2f' % Decimal(pars[0])
    peak_uncertainty = '%.2f' % Decimal(np.sqrt(cov[0][0]))
    residual = '%.2f' % abs(line - float(peak))

    label_01 = 'Peak = '+str(peak)+r' $\pm$ '+str(peak_uncertainty)
    label_02 = 'FWHM = '+str(FWHM)+r' $\pm$ '+str(FWHM_uncertainty)
    labels = [label_01, label_02,]

    plt.xlim(line_min,line_max)
    plt.xlabel('Energy (keV)', ha='right', x=1.0)
    plt.ylabel('Counts', ha='right', y=1.0)

    plt.tight_layout()
    plt.hist(df['e_cal'],range=(line_min,line_max), bins=nbin)
    plt.legend(labels, frameon=False, loc='upper right', fontsize='small')
    
    return peak, FWHM

if __name__ == '__main__':
        main()
