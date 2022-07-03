import pandas as pd
import sys
import numpy as np
import scipy as sp
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
plt.style.use('style.mplstyle')
np.set_printoptions(threshold=np.inf)

def main():

    # plug the values of the residuals from the results of fit_calibrated_peaks.py and fit_bkg_peaks.py to compare first-pass and second-pass calibrations.

    energies = [2614.51, 1764.49, 1460.82, 609.32, 351.93] 

    first_pass_residuals = [7.49, 3.14, 1.91, 0.73, 1.64]
    first_pass_errors = [0.05, 0.07, 0.01, 0.06, 0.01]
    plt.errorbar(energies, first_pass_residuals, first_pass_errors, color='blue', fmt='o', markersize=3, capsize=3, label='first-pass calibration residuals')

    second_pass_residuals = [0.02, 0.04, 0.00, 0.01, 0.00]
    second_pass_errors = [0.05, 0.05, 0.01, 0.04, 0.01]
    plt.errorbar(energies, second_pass_residuals, second_pass_errors, color='black', fmt='o', markersize=3, capsize=3, label='second-pass calibration residuals')

    plt.axhline(y=0.0, color='yellow', linestyle='-')
    plt.xlim(0,3000)
    plt.xlabel('Energy (keV)', ha='right', x='1.0')
    plt.ylabel('Calibration Fit Residuals', ha='right', y=1.0)
    plt.title('Krypton Data')
    plt.legend(frameon=True, loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
        main()

