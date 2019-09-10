import pandas as pd
import sys
import time
import numpy as np
import scipy as sp
import scipy.optimize as opt
import scipy.signal as signal
from scipy.signal import medfilt, find_peaks
import os, json
import pygama.dataset as ds
import pygama.analysis.histograms as pgh
import pygama.dsp.transforms as pgt
import pygama.utils as pgu
import pygama.analysis.peak_fitting as pga
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
plt.style.use('style.mplstyle')

def main():

    #linear_calibration()
    BKG_Subtraction()
    #dADC()

def linear_calibration():

    if(len(sys.argv) != 2):
        print('Usage: campaign.py [run number]')
        sys.exit()

    pks_lit = [609.3, 1460.8]

    with open("runDB.json") as f:
        runDB = json.load(f)
    tier_dir = os.path.expandvars(runDB["tier_dir"])
    meta_dir = os.path.expandvars(runDB["meta_dir"])

    df = pd.read_hdf('{}/t2_run{}.h5'.format(tier_dir,sys.argv[1]))
    print(df.keys())
    exit()
    m = np.array(df['e_ftp'])

    xlo, xhi, xpb = 0, 10000, 10
    nbins = int((xhi-xlo)/xpb)

    hist, bins = np.histogram(m, nbins, (xlo, xhi))
    bins = bins + (bins[1] - bins[0])/2
    bins = bins[0:(len(bins)-1)]

    hmed = medfilt(hist, 5)
    hpks = hist - hmed

    thresholds = np.arange(50,1000000,10, dtype=int)

    for i in range(len(thresholds)):
        maxes, mins = pgu.peakdet(hpks, thresholds[i], bins)
        if len(maxes) == 4:
            break

    x_maxes = []
    for i in range(len(maxes)):
        x_value = maxes[i][0]
        x_maxes.append(x_value)

    y_maxes = []
    for i in range(len(maxes)):
        y_value = maxes[i][1]
        y_maxes.append(y_value)


    ## if for whatever reason the background calibration does not seem to work, these values seem to work more often than not.
    #x_maxes = [15.0, 345.0, 585.0, 725.0, 835.0, 865.0, 1255.0, 1435.0, 1495.0, 1785.0, 2235.0, 2375.0, 2755.0, 3595.0, 4335.0, 6425.0]

    ratios = []
    for i in range(len(x_maxes)):
        for j in range(len(x_maxes)):
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
    #Now we will add a column to df that represents the energy measured (rather than only having the adc (e_ftp) value measured as the df currently does)

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
    plt.hist(n, np.arange(0,9500,0.75), histtype='step', color = 'black', zorder=2, label='{} entries'.format(len(n)))
    plt.xlim(0,4000)
    plt.ylim(0,plt.ylim()[1])
    plt.xlabel('Energy (keV)', ha='right', x=1.0)
    plt.ylabel('Counts', ha='right', y=1.0)
    E_cal = 'calibrated energy spectrum, run '+str(sys.argv[1])
    E_1 = 'E=238.6 keV (212Pb peak)'
    E_2 = 'E=351.93 keV (214Pb peak)'
    E_3 = 'E=511.0 keV (beta+ peak)'
    E_4 = 'E=583.19 keV (208Tl peak)'
    E_5 = 'E=609.32 keV (214Bi peak)*'
    E_6 = 'E=911.2 keV (228Ac peak)'
    E_7 = 'E=969 keV (228Ac peak)'
    E_8 = 'E=1120.3 keV (214Bi peak)'
    E_9 = 'E=1460.82 keV (40K peak)*'
    E_10 = 'E=1764.49 keV (214Bi peak)'
    E_11 = 'E=2614.51 keV (208Tl peak)'
    colors = ['black', 'red', 'aqua', 'darkgreen', 'darkorange', 'gray', 'brown', 'saddlebrown', 'navy', 'olive', 'indigo', 'limegreen']
    lines = [Line2D([0], [0], color=c) for c in colors]
    labels = [E_cal, E_1, E_2, E_3, E_4, E_5, E_6, E_7, E_8, E_9, E_10, E_11]
    #plt.title('Energy Spectrum')
    plt.legend(lines, labels, frameon=True, loc='upper right', fontsize='x-small')
    plt.tight_layout()
    #plt.semilogy()
    plt.show()

def BKG_Subtraction():

    if(len(sys.argv) != 3):
        print('Usage: campaign.py [background run number] [kryton run number]')
        sys.exit()

    with open("runDB.json") as f:
        runDB = json.load(f)
    meta_dir = os.path.expandvars(runDB["meta_dir"])

    BKG1 =  pd.read_hdf("{}/Spectrum_{}.hdf5".format(meta_dir,sys.argv[1]))
    Kr1 = pd.read_hdf("{}/Spectrum_{}.hdf5".format(meta_dir,sys.argv[2]))

    xlo, xhi, xpb = -0.25, 3000.25, 0.5
    nbins = int((xhi - xlo)/xpb)

    BKGhist, bins = np.histogram(BKG1['e_cal'], nbins, (xlo,xhi))
    Krhist, bins = np.histogram(Kr1['e_cal'], nbins, (xlo,xhi))

    bins = bins[0:(len(bins)-1)]
    bin_centers = bins + xpb/2

    integral1 = xpb * sum(BKGhist[50:2650])
    integral2 = xpb * sum(Krhist[50:2650])

    hist_01 = BKGhist * integral2/integral1
    hist3 = Krhist - hist_01
    #errors = np.sqrt(Krhist + hist_01*integral2/integral1)
    sigma_integral1 = xpb*np.sqrt(sum(BKGhist[50:2650]))
    sigma_integral2 = xpb*np.sqrt(sum(Krhist[50:2650]))
    errors = np.sqrt(Krhist+(integral2/integral1)**2*BKGhist+(BKGhist/integral1)**2*sigma_integral2**2+(BKGhist*integral2/integral1**2)**2*sigma_integral1**2)

    for i in range(len(hist3)):
        print('E = {} : {} +/- {}'.format(bin_centers[i],hist_01[i],errors[i]))

    #plt.plot(bin_centers, hist_01, color='red', ls='steps', label='Background Data')
    #plt.plot(bin_centers, Krhist, color='black', ls='steps', label='Kr83m Data')
    plt.plot(bin_centers, hist3/hist_01, color='black', ls='steps', label='Percent Error [(Kr - BKG)/BKG]')
    #plt.errorbar(bins, hist3, errors, color='black', fmt='o', markersize=3, capsize=3, label='Kr83m Spectrum Data Points')
    #plt.hlines(0, 0, 2650, color='red', linestyles='solid')

    plt.xlim(42,3500)
    #plt.ylim(-875,1000)
    plt.xlabel('Energy (keV)', ha='right', x=1.0)
    plt.ylabel('Residuals', ha='right', y=1.0)
    plt.title('Background Subtraction Percent Error')
    plt.legend(frameon=True, loc='best', fontsize='small')
    plt.tight_layout()
    #plt.semilogy()
    plt.show()

def dADC():

    if(len(sys.argv) != 2):
        print('Usage: campaign.py [run number]')
        sys.exit()

    with open("runDB.json") as f:
        runDB = json.load(f)
    meta_dir = os.path.expandvars(runDB["meta_dir"])

    df = pd.read_hdf('{}/Spectrum_{}.hdf5'.format(meta_dir,sys.argv[1]))

    plt.hist2d(df['e_cal'], df['dADC'], np.arange(-5,1000,0.2), norm=LogNorm())
    plt.xlim(0,500)
    plt.ylim(-5,100)
    plt.xlabel('Energy (keV)', ha='right', x=1.0)
    plt.ylabel('dADC', ha='right', y=1.0)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
        main()

