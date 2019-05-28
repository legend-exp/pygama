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

    # plug the values of the FWHM from the results of fit_calibrated_peaks.py to fit the detector resolution.

    FWHM = [3.79, 3.83, 4.52, 6.31]
    energies = [583.2, 911.2, 1460.8, 2614.5] 

    errors = [0.15, 0.21, 0.24, 0.28]
    plt.errorbar(energies, FWHM, errors, color='black', fmt='o', markersize=3, capsize=3, label='FWHM of BKG Peaks') 

    def func(x, a, b, c):
        return np.sqrt(a**2 + (b**2)*x + (c**2)*(x**2))

    optimizedParameters, pcov = opt.curve_fit(func, energies, FWHM)
    a_fit_value = '%.2f' % Decimal(optimizedParameters[0])
    b_fit_value = '%.2f' % Decimal(optimizedParameters[1])
    c_fit_value = '%.2f' % Decimal(optimizedParameters[2])

    print(a_fit_value)
    print(b_fit_value)
    print(c_fit_value)   
    print(np.sqrt(pcov[0][0]))
    print(np.sqrt(pcov[1][1]))
    print(np.sqrt(pcov[2][2])) 

    # find reduced chi_2 value
    chi_2_element_list = []
    for i in range(len(energies)):
        chi_2_element = abs((func(energies[i], *optimizedParameters) - FWHM[i])**2/func(energies[i], *optimizedParameters))
        chi_2_element_list.append(chi_2_element)

    chi_2 = sum(chi_2_element_list)
    reduced_chi_2 = '%.3e' % Decimal(chi_2/len(energies))
    
    x_vals = np.arange(0,4000,1)
    plt.plot(x_vals, func(x_vals, *optimizedParameters), color='darkorange', label=r'FWHM Fit $= \sqrt{a^2+b^2E+c^2E^2}$')
    #plt.scatter(energies, resolutions, color='black')
    plt.xlim(0,3000)
    plt.xlabel('Energy (keV)', ha='right', x='1.0')
    plt.ylabel('FWHM', ha='right', y=1.0)
    plt.title('FWHM vs. E')
    plt.legend(frameon=True, loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
        main()

