import pandas as pd
import sys
import json
import os
import numpy as np
import scipy as sp
from scipy.signal import medfilt, find_peaks
import pygama.analysis.histograms as pgh
import pygama.analysis.peak_fitting as pga
import matplotlib.pyplot as plt

from pygama import DataSet
from pygama.analysis.calibration import *
from pygama.analysis.histograms import *
import pygama.utils as pgu
from matplotlib.lines import Line2D
from pygama.utils import set_plot_style
# plt.style.use('style.mplstyle')
np.set_printoptions(threshold=np.inf)

def main():

    if(len(sys.argv) != 2):
        print('Usage: thorium_calibration.py [run number]')
        sys.exit()

    #spectrum_medfilt_peaks()
    linear_calibration()

def spectrum_medfilt_peaks():

    with open("runDB.json") as f:
        runDB = json.load(f)
    tier_dir = os.path.expandvars(runDB["tier_dir"])
    meta_dir = os.path.expandvars(runDB["meta_dir"])

    df = pd.read_hdf('{}/t2_run{}.h5'.format(tier_dir,sys.argv[1]))

    m = np.array(df['e_ftp'])

    xlo, xhi, xpb = 0, 10000, 1
    nbins = int((xhi-xlo)/xpb)

    hist, bins = np.histogram(m, nbins, (xlo, xhi))
    hist = np.pad(hist, (1,0), 'constant')
    bins = bins + (bins[1] - bins[0])/2

    hmed = medfilt(hist, 51)
    hpks = hist - hmed

    plt.plot(bins, hist, '-k', ls="steps", label='uncalibrated energy spectrum')
    plt.plot(bins, hmed, '-r', ls='steps', label="peakless spectrum (medfilt)")
    plt.plot(bins, hpks, '-b', ls='steps', label='peaks only (spectrum - medfilt)')

    thresholds = np.arange(5,10000,5, dtype=int)

    for i in range(len(thresholds)):
        maxes, mins = pgu.peakdet(hpks, thresholds[i], bins)
        if len(maxes) == 5:
            break

    for pk in maxes:
        plt.plot(pk[0],pk[1], '.m', ms=10)

    plt.xlim(0,9000)
    plt.ylim(0,plt.ylim()[1])
    colors = ['black', 'red', 'blue']
    lines = [Line2D([0], [0], color=c) for c in colors]
    labels = ['uncalibrated energy spectrum', 'peakless spectrum (medfilt)', 'peaks only (spectrum - medfilt)']
    plt.legend(lines, labels, frameon=True, loc='upper right', fontsize='x-small')
    plt.show()

def linear_calibration():

    with open("runDB.json") as f:
        runDB = json.load(f)
    tier_dir = os.path.expandvars(runDB["tier_dir"])
    meta_dir = os.path.expandvars(runDB["meta_dir"])

    pks_lit = [238.6, 583.2]

    pks_lit = [238.6, 583.2]

    df = pd.read_hdf('{}/t2_run{}.h5'.format(tier_dir,sys.argv[1]))
    # ds = DataSet(runlist=[278, 279], md='./runDB.json', tier_dir=tier_dir)
    # df = ds.get_t2df()
    m = np.array(df['e_ftp'])

    xlo, xhi, xpb = 0, 10000, 1
    nbins = int((xhi-xlo)/xpb)

    hist, bins = np.histogram(m, nbins, (xlo, xhi))
    hist = np.pad(hist, (1,0), 'constant')
    bins = bins + (bins[1] - bins[0])/2

    hmed = medfilt(hist, 51)
    hpks = hist - hmed

    thresholds = np.arange(5,10000,5, dtype=int)

    for i in range(len(thresholds)):
        maxes, mins = pgu.peakdet(hpks, thresholds[i], bins)
        if len(maxes) == 5:
            break

    x_maxes = []
    for i in range(len(maxes)):
        x_value = maxes[i][0]
        x_maxes.append(x_value)

    ratios = []
    for i in range(len(maxes)):
        for j in range(len(maxes)):
            ratios.append(x_maxes[i]/x_maxes[j])

    #another way to do the block above
    #import itertools
    #ratios = []
    #for x_i, x_j in itertools.product(x_maxes, x_maxes):
        #ratios.append(x_i/x_j)

    real_ratio = []
    for i in range(len(ratios)):
        real_ratio.append(pks_lit[1]/pks_lit[0])

    ratios_array = np.array(ratios)
    real_ratio_array = np.array(real_ratio)

    closeness = np.absolute(ratios_array - real_ratio_array)

    relevant_entry = int(np.where(closeness == np.amin(closeness))[0])

    adc_2_peak_combinations = []
    for i in range(len(x_maxes)):
        for j in range(len(x_maxes)):
            adc_2_peak_combinations.append([x_maxes[j], x_maxes[i]])

    #ADC Values Corresponding to Energy Peaks 1460.820 keV and 2614.511 keV
    adc_values = adc_2_peak_combinations[relevant_entry]

    #Now we model a linear equation to go from ADC value (e_ftp) to real energy using the points (adc_values[0], 1460.820) and (adc_values[1], 2614.511 keV)
    # E = A(e_ftp) + B
    A = float((pks_lit[1] - pks_lit[0])/(adc_values[1]-adc_values[0]))
    B = float((pks_lit[1] - adc_values[1]*A))
    #Now we will add a column to df that represents the energy measured (rather than only having the adc (e_ftp) value measured as the df currently does)
    df['e_cal'] = df['e_ftp']*A+B

    pks_lit_all = [238.6, 338.3, 463.0, 511,0, 583.2, 727.3, 794.9, 860.6, 911.2, 969, 1460.8, 1592.5, 2103.5, 2614.5]
    plt.axvline(x=238.6, ymin=0, ymax=30, color='red', linestyle='--', lw=1, zorder=1)
    plt.axvline(x=338.3, ymin=0, ymax=30, color='aqua', linestyle='--', lw=1, zorder=1)
    plt.axvline(x=463.0, ymin=0, ymax=30, color='teal', linestyle='--', lw=1, zorder=1)
    plt.axvline(x=511.0, ymin=0, ymax=30, color='darkgreen', linestyle='--', lw=1, zorder=1)
    plt.axvline(x=583.2, ymin=0, ymax=30, color='darkorange', linestyle='--', lw=1, zorder=1)
    plt.axvline(x=727.3, ymin=0, ymax=30, color='gray', linestyle='--', lw=1, zorder=1)
    plt.axvline(x=794.9, ymin=0, ymax=30, color='brown', linestyle='--', lw=1, zorder=1)
    plt.axvline(x=860.6, ymin=0, ymax=30, color='purple', linestyle='--', lw=1, zorder=1)
    plt.axvline(x=911.2, ymin=0, ymax=30, color='fuchsia', linestyle='--', lw=1, zorder=1)
    plt.axvline(x=969.0, ymin=0, ymax=30, color='saddlebrown', linestyle='--', lw=1, zorder=1)
    plt.axvline(x=1460.8, ymin=0, ymax=30, color='navy', linestyle='--', lw=1, zorder=1)
    plt.axvline(x=1592.5, ymin=0, ymax=30, color='limegreen', linestyle='--', lw=1, zorder=1)
    plt.axvline(x=2103.5, ymin=0, ymax=30, color='olive', linestyle='--', lw=1, zorder=1)
    plt.axvline(x=2614.5, ymin=0, ymax=30, color='indigo', linestyle='--', lw=1, zorder=1)
    n = np.array(df['e_cal'])
    plt.hist(n, np.arange(0,9500,1.5), histtype='step', color = 'black', zorder=2, label='{} entries'.format(len(n)))
    plt.xlim(0,4000)
    plt.ylim(0,plt.ylim()[1])
    plt.xlabel('Energy (keV)', ha='right', x=1.0)
    plt.ylabel('Counts', ha='right', y=1.0)
    E_cal = 'calibrated energy spectrum, run '+str(sys.argv[1])
    E_1 = 'E=238.6 keV (212Pb peak)*'
    E_2 = 'E=338.3 keV (228Ac peak)'
    E_3 = 'E=463.0 keV (228Ac peak)'
    E_4 = 'E=511.0 keV (beta+ peak)'
    E_5 = 'E=583.2 keV (208Tl peak)*'
    E_6 = 'E=727.3 keV (212Bi peak)'
    E_7 = 'E=794.9 keV (228Ac peak)'
    E_8 = 'E=860.6 keV (208Tl peak)'
    E_9 = 'E=911.2 keV (228Ac peak)'
    E_10 = 'E=969 keV (228Ac peak)'
    E_11 = 'E=1460.8 keV (40K peak)'
    E_12 = 'E=1592.5 keV (208Tl DE)'
    E_13 = 'E=2103.5 keV (208Tl SE)'
    E_14 = 'E=2614.5 keV (208Tl peak)'
    colors = ['black', 'red', 'aqua', 'teal', 'darkgreen', 'darkorange', 'gray', 'brown', 'purple', 'fuchsia', 'saddlebrown', 'navy', 'limegreen', 'olive', 'indigo']
    lines = [Line2D([0], [0], color=c) for c in colors]
    labels = [E_cal, E_1, E_2, E_3, E_4, E_5, E_6, E_7, E_8, E_9, E_10, E_11, E_12, E_13, E_14]
    plt.legend(lines, labels, frameon=True, loc='upper right', fontsize='x-small')
    plt.tight_layout()
    #plt.semilogy()
    plt.show()

if __name__ == '__main__':
        main()
