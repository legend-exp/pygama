import pandas as pd
import json
import os
import sys
import numpy as np
import scipy as sp
import scipy.optimize as opt
from scipy.signal import medfilt, find_peaks
from pygama import DataSet
import pygama.analysis.histograms as pgh
import pygama.utils as pgu
import pygama.analysis.peak_fitting as pga
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# plt.style.use('style.mplstyle')
np.set_printoptions(threshold=np.inf)

def main():

    if(len(sys.argv) != 2):
        print('Usage: bkg_calibration.py [run number]')
        sys.exit()

    plot_raw()
    #spectrum_medfilt_peaks()
    #linear_calibration()

def plot_raw():

    with open("runDB.json") as f:
        runDB = json.load(f)
    tier_dir = os.path.expandvars(runDB["tier_dir"])
    meta_dir = os.path.expandvars(runDB["meta_dir"])

    df = pd.read_hdf('{}/t2_run{}.h5'.format(tier_dir,sys.argv[1]))

    m = np.array(df['e_ftp'])

    plt.hist(m, np.arange(0,9500,0.5), histtype='step', color = 'black', label='non-calibrated spectrum')
    plt.xlabel('e_ftp', ha='right', x=1.0)
    plt.ylabel('Counts', ha='right', y=1.0)
    plt.xlim(0,9500)
    #plt.ylim(0,4000)
    plt.legend(frameon=True, loc='upper right', fontsize='small')
    plt.show()

def spectrum_medfilt_peaks():

    with open("runDB.json") as f:
        runDB = json.load(f)
    tier_dir = os.path.expandvars(runDB["tier_dir"])
    meta_dir = os.path.expandvars(runDB["meta_dir"])

    df = pd.read_hdf('{}/t2_run{}.h5'.format(tier_dir,sys.argv[1]))

    m = np.array(df['e_ftp'])

    xlo, xhi, xpb = 0, 10000, 10
    nbins = int((xhi-xlo)/xpb)

    hist, bins = np.histogram(m, nbins, (xlo, xhi))
    #bins = bins + (bins[1] - bins[0])/2
    #bins = bins[0:(len(bins)-1)]
    hist = np.append(hist, 0)

    hmed = medfilt(hist, 5)
    hpks = hist - hmed

    plt.plot(bins, hist, '-k', ls="steps", label='uncalibrated energy spectrum, run '+str(sys.argv[1]))
    plt.plot(bins, hmed, '-r', ls='steps', label="peakless spectrum (medfilt)")
    plt.plot(bins, hpks, '-b', ls='steps', label='peaks only (spectrum - medfilt)')

    thresholds = np.arange(5,10000,5, dtype=int)

    for i in range(len(thresholds)):
        maxes, mins = pgu.peakdet(hpks, thresholds[i], bins)
        if len(maxes) == 4:
            break

    for pk in maxes:
        plt.plot(pk[0],pk[1], '.m', ms=10)

    plt.xlim(0,9000)
    plt.ylim(0,plt.ylim()[1])
    #plt.semilogy()
    colors = ['black', 'red', 'blue']
    lines = [Line2D([0], [0], color=c) for c in colors]
    labels = ['uncalibrated energy spectrum, run '+str(sys.argv[1]), 'peakless spectrum (medfilt)', 'peaks only (spectrum - medfilt)']
    plt.legend(lines, labels, frameon=True, loc='upper right', fontsize='x-small')
    plt.show()


def linear_calibration():

    pks_lit = [609.32, 1460.82]

    with open("runDB.json") as f:
        runDB = json.load(f)
    tier_dir = os.path.expandvars(runDB["tier_dir"])
    meta_dir = os.path.expandvars(runDB["meta_dir"])

    df = pd.read_hdf('{}/t2_run{}.h5'.format(tier_dir,sys.argv[1]))

    m = np.array(df['e_ftp'])

    xlo, xhi, xpb = 0, 10000, 10
    nbins = int((xhi-xlo)/xpb)

    hist, bins = np.histogram(m, nbins, (xlo, xhi))
    bins = bins + (bins[1] - bins[0])/2
    bins = bins[0:(len(bins)-1)]
    #hist = np.append(hist, 0)

    hmed = medfilt(hist, 5)
    hpks = hist - hmed

    thresholds = np.arange(50,750,10, dtype=int)

    for i in range(len(thresholds)):
        maxes, mins = pgu.peakdet(hpks, thresholds[i], bins)
        if len(maxes) == 4:
            break

    x_maxes = []
    for i in range(len(maxes)):
        x_value = maxes[i][0]
        x_maxes.append(x_value)

    ratios = []
    for i in range(len(maxes)):
        for j in range(len(maxes)):
            ratios.append(x_maxes[i]/x_maxes[j])

    print(x_maxes)    

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

    relevant_entry = np.where(closeness == np.amin(closeness))[0]
    relevant_entry = int(relevant_entry[len(relevant_entry)-1])

    adc_2_peak_combinations = []
    for i in range(len(x_maxes)):
        for j in range(len(x_maxes)):
            adc_2_peak_combinations.append([x_maxes[j], x_maxes[i]])

    #ADC Values Corresponding to Energy Peaks 1460.820 keV and 2614.511 keV
    adc_values = adc_2_peak_combinations[relevant_entry]

    #Now we model a linear equation to go from ADC value (e_ftp) to real energy using the points (adc_values[0], 1460.820) and (adc_values[1], 2614.511 keV)
    # E = A(e_ftp) + B
    A = float((pks_lit[1]-pks_lit[0])/(adc_values[1]-adc_values[0]))
    B = float((pks_lit[1] - adc_values[1]*A))
    A = 0.4074162679425837
    B = 0.23267942583743206
    #Now we will add a column to df that represents the energy measured (rather than only having the adc (e_ftp) value measured as the df currently does)
    print('E = {}(e_ftp) + {}'.format(A,B))
    print(A)
    print(B)

    df["e_cal"] = A * df['e_ftp'] + B

    df.to_hdf('{}/Spectrum_{}.hdf5'.format(meta_dir,sys.argv[1]), key='df', mode='w')   

    pks_lit_all = [238.6, 351.9, 511.0, 583.2, 609.3, 911.2, 969, 1120.3, 1460.8, 1764.5, 2614.5]
    plt.axvline(x=238.6, ymin=0, ymax=30, color='red', linestyle='--', lw=1, zorder=1)
    plt.axvline(x=351.9, ymin=0, ymax=30, color='aqua', linestyle='--', lw=1, zorder=1)
    plt.axvline(x=511.0, ymin=0, ymax=30, color='darkgreen', linestyle='--', lw=1, zorder=1)
    plt.axvline(x=583.2, ymin=0, ymax=30, color='darkorange', linestyle='--', lw=1, zorder=1)
    plt.axvline(x=609.3, ymin=0, ymax=30, color='gray', linestyle='--', lw=1, zorder=1)
    plt.axvline(x=911.2, ymin=0, ymax=30, color='brown', linestyle='--', lw=1, zorder=1)
    plt.axvline(x=969.0, ymin=0, ymax=30, color='saddlebrown', linestyle='--', lw=1, zorder=1)
    plt.axvline(x=1120.3, ymin=0, ymax=30, color='navy', linestyle='--', lw=1, zorder=1)
    plt.axvline(x=1460.8, ymin=0, ymax=30, color='olive', linestyle='--', lw=1, zorder=1)
    plt.axvline(x=1764.5, ymin=0, ymax=30, color='indigo', linestyle='--', lw=1, zorder=1)
    plt.axvline(x=2614.5, ymin=0, ymax=30, color='limegreen', linestyle='--', lw=1, zorder=1)
    n = np.array(df['e_cal'])
    plt.hist(n, np.arange(0,9500,0.5), histtype='step', color = 'black', zorder=2, label='{} entries'.format(len(n)))
    plt.xlim(0,4000)
    plt.ylim(0,plt.ylim()[1])
    plt.xlabel('Energy (keV)', ha='right', x=1.0)
    plt.ylabel('Counts', ha='right', y=1.0)
    E_cal = 'calibrated energy spectrum, run '+str(sys.argv[1])
    E_1 = 'E=238.6 keV (212Pb peak)'
    E_2 = 'E=351.9 keV (214Pb peak)'
    E_3 = 'E=511.0 keV (beta+ peak)'
    E_4 = 'E=583.2 keV (208Tl peak)'
    E_5 = 'E=609.3 keV (214Bi peak)*'
    E_6 = 'E=911.2 keV (228Ac peak)'
    E_7 = 'E=969 keV (228Ac peak)'
    E_8 = 'E=1120.3 keV (214Bi peak)'
    E_9 = 'E=1460.8 keV (40K peak)*'
    E_10 = 'E=1764.5 keV (214Bi peak)'
    E_11 = 'E=2614.5 keV (208Tl peak)'
    colors = ['black', 'red', 'aqua', 'darkgreen', 'darkorange', 'gray', 'brown', 'saddlebrown', 'navy', 'olive', 'indigo', 'limegreen']
    lines = [Line2D([0], [0], color=c) for c in colors]
    labels = [E_cal, E_1, E_2, E_3, E_4, E_5, E_6, E_7, E_8, E_9, E_10, E_11]
    #plt.title('Energy Spectrum')
    plt.legend(lines, labels, frameon=True, loc='upper right', fontsize='x-small')
    plt.tight_layout()
    #plt.semilogy()
    plt.show()

if __name__ == '__main__':
        main()
