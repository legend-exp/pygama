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


def main():

    ## this code takes the peaks from thorium's first-pass calibration and fits them. the values from these fits are used to then do a non-linear, second-pass calibration.

    #peak_2615()
    peak_1460()
    #peak_969()
    #peak_911()
    peak_583()
    #peak_511()
    #peak_239()

def peak_2615():
    
    df =  pd.read_hdf("Spectrum_203.hdf5", key="df")

    def gauss(x, mu, sigma, A=1):
        """
        define a gaussian distribution, w/ args: mu, sigma, area (optional).
        """
        return A * (1. / sigma / np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2. * sigma**2))

    def radford_peak(x, mu, sigma, hstep, htail, tau, bg0, a=1):
        """
        David Radford's HPGe peak shape function
        """
        # make sure the fractional amplitude parameters stay reasonable...
        if htail < 0 or htail > 1: 
            return np.zeros_like(x)
        if hstep < 0 or hstep > 1: 
            return np.zeros_like(x)

        bg_term = bg0  #+ x*bg1
        if np.any(bg_term < 0): 
            return np.zeros_like(x)

        # compute the step and the low energy tail
        step = a * hstep * erfc((x - mu) / (sigma * np.sqrt(2)))
        le_tail = a * htail
        le_tail *= erfc((x - mu) / (sigma * np.sqrt(2)) + sigma / (tau * np.sqrt(2)))
        le_tail *= np.exp((x - mu) / tau)
        le_tail /= (2 * tau * np.exp(-(sigma / (np.sqrt(2) * tau))**2))

        # add up all the peak shape components
        return (1 - htail) * gauss(x, mu, sigma, a) + bg_term + step + le_tail


    hist, bins, var = pgh.get_hist(df['e_cal'], range=(2590,2640), dx=0.5)
    pgh.plot_hist(hist, bins, var=hist, label="data")
    pars, cov = pga.fit_hist(radford_peak, hist, bins, var=hist, guess=[2614.5, 2.95, 0.001, 0.3, 5, 1, 900])
    #pgu.print_fit_results(pars, cov, radford_peak)
    pgu.plot_func(radford_peak, pars, label="chi2 fit", color='red')
    #x_vals = np.arange(2590,2640,0.5)
    #plt.plot(x_vals, radford_peak(x_vals, 2614.5, 2.95, .001, 0.3, 5, 1, 900))
    
    FWHM = '%.2f' % Decimal(pars[1]*2)
    FWHM_uncertainty = '%.2f' % Decimal(np.sqrt(cov[1][1])*2)
    peak = '%.2f' % Decimal(pars[0])
    peak_uncertainty = '%.2f' % Decimal(np.sqrt(cov[0][0]))
    
    chi_2_element_list = []
    for i in range(len(hist)):
        chi_2_element = abs((radford_peak(bins[i], *pars) - hist[i])**2/radford_peak(bins[i], *pars))
        chi_2_element_list.append(chi_2_element)
    chi_2 = sum(chi_2_element_list)
    reduced_chi_2 = '%.2f' % Decimal(chi_2/len(hist))

    label_01 = '2614.5 keV peak fit'
    label_02 = 'FWHM = '+str(FWHM)+r' $\pm$ '+str(FWHM_uncertainty)
    label_03 = 'Peak = '+str(peak)+r' $\pm$ '+str(peak_uncertainty)
    label_04 = r'Reduced $\chi^2 = $'+str(reduced_chi_2)
    colors = ['red', 'red','red','red']
    lines = [Line2D([0], [0], color=c, lw=2) for c in colors]
    labels = [label_01, label_02, label_03, label_04]
    
    plt.xlim(2590,2640)
    plt.ylim(0,plt.ylim()[1])
    plt.xlabel('Energy (keV)', ha='right', x=1.0)
    plt.ylabel('Counts', ha='right', y=1.0)

    plt.tight_layout()
    #plt.semilogy()
    plt.legend(lines, labels, frameon=False, loc='upper right', fontsize='small')
    plt.show()

def peak_1460():

    df =  pd.read_hdf("Spectrum_203.hdf5", key="df")

    def gauss(x, mu, sigma, A=1):
        """
        define a gaussian distribution, w/ args: mu, sigma, area (optional).
        """
        return A * (1. / sigma / np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2. * sigma**2))

    def radford_peak(x, mu, sigma, hstep, htail, tau, bg0, a=1):
        """
        David Radford's HPGe peak shape function
        """
        # make sure the fractional amplitude parameters stay reasonable...
        if htail < 0 or htail > 1:
            return np.zeros_like(x)
        if hstep < 0 or hstep > 1:
            return np.zeros_like(x)

        bg_term = bg0  #+ x*bg1
        if np.any(bg_term < 0):
            return np.zeros_like(x)

        # compute the step and the low energy tail
        step = a * hstep * erfc((x - mu) / (sigma * np.sqrt(2)))
        le_tail = a * htail
        le_tail *= erfc((x - mu) / (sigma * np.sqrt(2)) + sigma / (tau * np.sqrt(2)))
        le_tail *= np.exp((x - mu) / tau)
        le_tail /= (2 * tau * np.exp(-(sigma / (np.sqrt(2) * tau))**2))

        # add up all the peak shape components
        return (1 - htail) * gauss(x, mu, sigma, a) + bg_term + step + le_tail

    hist, bins, var = pgh.get_hist(df['e_cal'], range=(1445,1475), dx=0.5)
    pgh.plot_hist(hist, bins, var=hist, label="data")
    pars, cov = pga.fit_hist(radford_peak, hist, bins, var=hist, guess=[1460.8, 2.95, 0.0001, 0.03, 4, 1, 250])
    pgu.print_fit_results(pars, cov, radford_peak)
    pgu.plot_func(radford_peak, pars, label="chi2 fit", color='red')
    #x_vals = np.arange(1455,1470,0.5)
    #plt.plot(x_vals, radford_peak(x_vals, 1460.8, 2.95, .001, 0.03, 5, 1, 250))

    FWHM = '%.2f' % Decimal(pars[1]*2)
    FWHM_uncertainty = '%.2f' % Decimal(np.sqrt(cov[1][1])*2)
    peak = '%.2f' % Decimal(pars[0])
    peak_uncertainty = '%.2f' % Decimal(np.sqrt(cov[0][0]))

    chi_2_element_list = []
    for i in range(len(hist)):
        chi_2_element = abs((radford_peak(bins[i], *pars) - hist[i])**2/radford_peak(bins[i], *pars))
        chi_2_element_list.append(chi_2_element)
    chi_2 = sum(chi_2_element_list)
    reduced_chi_2 = '%.2f' % Decimal(chi_2/len(hist))

    label_01 = '1460.8 keV peak fit'
    label_02 = 'FWHM = '+str(FWHM)+r' $\pm$ '+str(FWHM_uncertainty)
    label_03 = 'Peak = '+str(peak)+r' $\pm$ '+str(peak_uncertainty)
    label_04 = r'Reduced $\chi^2 = $'+str(reduced_chi_2)
    colors = ['red', 'red','red','red']
    lines = [Line2D([0], [0], color=c, lw=2) for c in colors]
    labels = [label_01, label_02, label_03, label_04]

    plt.xlim(1445,1475)
    plt.ylim(0,plt.ylim()[1])
    plt.xlabel('Energy (keV)', ha='right', x=1.0)
    plt.ylabel('Counts', ha='right', y=1.0)

    plt.tight_layout()
    plt.legend(lines, labels, frameon=False, loc='upper right', fontsize='small')
    #plt.semilogy()
    plt.show()

def peak_969():

    df =  pd.read_hdf("Spectrum_203.hdf5", key="df")

    def gauss(x, mu, sigma, A=1):
        """
        define a gaussian distribution, w/ args: mu, sigma, area (optional).
        """
        return A * (1. / sigma / np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2. * sigma**2))

    def radford_peak(x, mu, sigma, hstep, htail, tau, bg0, a=1):
        """
        David Radford's HPGe peak shape function
        """
        # make sure the fractional amplitude parameters stay reasonable...
        if htail < 0 or htail > 1:
            return np.zeros_like(x)
        if hstep < 0 or hstep > 1:
            return np.zeros_like(x)

        bg_term = bg0  #+ x*bg1
        if np.any(bg_term < 0):
            return np.zeros_like(x)

        # compute the step and the low energy tail
        step = a * hstep * erfc((x - mu) / (sigma * np.sqrt(2)))
        le_tail = a * htail
        le_tail *= erfc((x - mu) / (sigma * np.sqrt(2)) + sigma / (tau * np.sqrt(2)))
        le_tail *= np.exp((x - mu) / tau)
        le_tail /= (2 * tau * np.exp(-(sigma / (np.sqrt(2) * tau))**2))

        # add up all the peak shape components
        return (1 - htail) * gauss(x, mu, sigma, a) + bg_term + step + le_tail

    hist, bins, var = pgh.get_hist(df['e_cal'], range=(930,1010), dx=0.5)
    pgh.plot_hist(hist, bins, var=hist, label="data")
    pars, cov = pga.fit_hist(radford_peak, hist, bins, var=hist, guess=[968, 2.35, 0.003, 0.07, 5, 8, 800])
    pgu.print_fit_results(pars, cov, radford_peak)
    pgu.plot_func(radford_peak, pars, label="chi2 fit", color='red')
    #x_vals = np.arange(940,1000,0.5)
    #plt.plot(x_vals, radford_peak(x_vals, 968, 2.35, .003, 0.07, 5, 8, 800))

    FWHM = '%.2f' % Decimal(pars[1]*2)
    FWHM_uncertainty = '%.2f' % Decimal(np.sqrt(cov[1][1])*2)
    peak = '%.2f' % Decimal(pars[0])
    peak_uncertainty = '%.2f' % Decimal(np.sqrt(cov[0][0]))

    chi_2_element_list = []
    for i in range(len(hist)):
        chi_2_element = abs((radford_peak(bins[i], *pars) - hist[i])**2/radford_peak(bins[i], *pars))
        chi_2_element_list.append(chi_2_element)
    chi_2 = sum(chi_2_element_list)
    reduced_chi_2 = '%.2f' % Decimal(chi_2/len(hist))

    label_01 = '969 keV peak fit'
    label_02 = 'FWHM = '+str(FWHM)+r' $\pm$ '+str(FWHM_uncertainty)
    label_03 = 'Peak = '+str(peak)+r' $\pm$ '+str(peak_uncertainty)
    label_04 = r'Reduced $\chi^2 = $'+str(reduced_chi_2)
    colors = ['red', 'red','red','red']
    lines = [Line2D([0], [0], color=c, lw=2) for c in colors]
    labels = [label_01, label_02, label_03, label_04]

    plt.xlim(930,1010)
    plt.ylim(0,plt.ylim()[1])
    plt.xlabel('Energy (keV)', ha='right', x=1.0)
    plt.ylabel('Counts', ha='right', y=1.0)

    plt.tight_layout()
    plt.legend(lines, labels, frameon=False, loc='upper right', fontsize='small')
    #plt.semilogy()
    plt.show()

def peak_911():

    df =  pd.read_hdf("Spectrum_203.hdf5", key="df")

    def gauss(x, mu, sigma, A=1):
        """
        define a gaussian distribution, w/ args: mu, sigma, area (optional).
        """
        return A * (1. / sigma / np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2. * sigma**2))

    def radford_peak(x, mu, sigma, hstep, htail, tau, bg0, a=1):
        """
        David Radford's HPGe peak shape function
        """
        # make sure the fractional amplitude parameters stay reasonable...
        if htail < 0 or htail > 1:
            return np.zeros_like(x)
        if hstep < 0 or hstep > 1:
            return np.zeros_like(x)

        bg_term = bg0  #+ x*bg1
        if np.any(bg_term < 0):
            return np.zeros_like(x)

        # compute the step and the low energy tail
        step = a * hstep * erfc((x - mu) / (sigma * np.sqrt(2)))
        le_tail = a * htail
        le_tail *= erfc((x - mu) / (sigma * np.sqrt(2)) + sigma / (tau * np.sqrt(2)))
        le_tail *= np.exp((x - mu) / tau)
        le_tail /= (2 * tau * np.exp(-(sigma / (np.sqrt(2) * tau))**2))

        # add up all the peak shape components
        return (1 - htail) * gauss(x, mu, sigma, a) + bg_term + step + le_tail

    hist, bins, var = pgh.get_hist(df['e_cal'], range=(890,930), dx=0.5)
    pgh.plot_hist(hist, bins, var=hist, label="data")
    pars, cov = pga.fit_hist(radford_peak, hist, bins, var=hist, guess=[911.1, 1.95, 0.003, 0.07, 5, 8, 1010])
    pgu.print_fit_results(pars, cov, radford_peak)
    pgu.plot_func(radford_peak, pars, label="chi2 fit", color='red')
    #x_vals = np.arange(890,930,0.5)
    #plt.plot(x_vals, radford_peak(x_vals, 911.1, 1.95, .003, 0.07, 5, 8, 1010))

    FWHM = '%.2f' % Decimal(pars[1]*2)
    FWHM_uncertainty = '%.2f' % Decimal(np.sqrt(cov[1][1])*2)
    peak = '%.2f' % Decimal(pars[0])
    peak_uncertainty = '%.2f' % Decimal(np.sqrt(cov[0][0]))

    chi_2_element_list = []
    for i in range(len(hist)):
        chi_2_element = abs((radford_peak(bins[i], *pars) - hist[i])**2/radford_peak(bins[i], *pars))
        chi_2_element_list.append(chi_2_element)
    chi_2 = sum(chi_2_element_list)
    reduced_chi_2 = '%.2f' % Decimal(chi_2/len(hist))

    label_01 = '911.2 keV peak fit'
    label_02 = 'FWHM = '+str(FWHM)+r' $\pm$ '+str(FWHM_uncertainty)
    label_03 = 'Peak = '+str(peak)+r' $\pm$ '+str(peak_uncertainty)
    label_04 = r'Reduced $\chi^2 = $'+str(reduced_chi_2)
    colors = ['red', 'red','red','red']
    lines = [Line2D([0], [0], color=c, lw=2) for c in colors]
    labels = [label_01, label_02, label_03, label_04]

    plt.xlim(890,930)
    plt.ylim(0,plt.ylim()[1])
    plt.xlabel('Energy (keV)', ha='right', x=1.0)
    plt.ylabel('Counts', ha='right', y=1.0)

    plt.tight_layout()
    plt.legend(lines, labels, frameon=False, loc='upper right', fontsize='small')
    #plt.semilogy()
    plt.show()

def peak_583():

    df =  pd.read_hdf("Spectrum_203.hdf5", key="df")

    def gauss(x, mu, sigma, A=1):
        """
        define a gaussian distribution, w/ args: mu, sigma, area (optional).
        """
        return A * (1. / sigma / np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2. * sigma**2))

    def radford_peak(x, mu, sigma, hstep, htail, tau, bg0, a=1):
        """
        David Radford's HPGe peak shape function
        """
        # make sure the fractional amplitude parameters stay reasonable...
        if htail < 0 or htail > 1:
            return np.zeros_like(x)
        if hstep < 0 or hstep > 1:
            return np.zeros_like(x)

        bg_term = bg0  #+ x*bg1
        if np.any(bg_term < 0):
            return np.zeros_like(x)

        # compute the step and the low energy tail
        step = a * hstep * erfc((x - mu) / (sigma * np.sqrt(2)))
        le_tail = a * htail
        le_tail *= erfc((x - mu) / (sigma * np.sqrt(2)) + sigma / (tau * np.sqrt(2)))
        le_tail *= np.exp((x - mu) / tau)
        le_tail /= (2 * tau * np.exp(-(sigma / (np.sqrt(2) * tau))**2))

        # add up all the peak shape components
        return (1 - htail) * gauss(x, mu, sigma, a) + bg_term + step + le_tail

    hist, bins, var = pgh.get_hist(df['e_cal'], range=(560,600), dx=0.5)
    pgh.plot_hist(hist, bins, var=hist, label="data")
    pars, cov = pga.fit_hist(radford_peak, hist, bins, var=hist, guess=[583.2, 1.89, 0.005, 0.1, 2, 34, 1171])
    pgu.print_fit_results(pars, cov, radford_peak)
    pgu.plot_func(radford_peak, pars, label="chi2 fit", color='red')
    #x_vals = np.arange(560,600,0.5)
    #plt.plot(x_vals, radford_peak(x_vals, 583.2, 1.89, .005, 0.1, 2, 34, 1171))

    FWHM = '%.2f' % Decimal(pars[1]*2)
    FWHM_uncertainty = '%.2f' % Decimal(np.sqrt(cov[1][1])*2)
    peak = '%.2f' % Decimal(pars[0])
    peak_uncertainty = '%.2f' % Decimal(np.sqrt(cov[0][0]))

    chi_2_element_list = []
    for i in range(len(hist)):
        chi_2_element = abs((radford_peak(bins[i], *pars) - hist[i])**2/radford_peak(bins[i], *pars))
        chi_2_element_list.append(chi_2_element)
    chi_2 = sum(chi_2_element_list)
    reduced_chi_2 = '%.2f' % Decimal(chi_2/len(hist))

    label_01 = '583.2 keV peak fit'
    label_02 = 'FWHM = '+str(FWHM)+r' $\pm$ '+str(FWHM_uncertainty)
    label_03 = 'Peak = '+str(peak)+r' $\pm$ '+str(peak_uncertainty)
    label_04 = r'Reduced $\chi^2 = $'+str(reduced_chi_2)
    colors = ['red', 'red','red','red']
    lines = [Line2D([0], [0], color=c, lw=2) for c in colors]
    labels = [label_01, label_02, label_03, label_04]

    plt.xlim(560,600)
    plt.ylim(0,plt.ylim()[1])
    plt.xlabel('Energy (keV)', ha='right', x=1.0)
    plt.ylabel('Counts', ha='right', y=1.0)

    plt.tight_layout()
    plt.legend(lines, labels, frameon=False, loc='upper right', fontsize='small')
    #plt.semilogy()
    plt.show()

def peak_511():

    df =  pd.read_hdf("Spectrum_203.hdf5", key="df")

    def gauss(x, mu, sigma, A=1):
        """
        define a gaussian distribution, w/ args: mu, sigma, area (optional).
        """
        return A * (1. / sigma / np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2. * sigma**2))

    def radford_peak(x, mu, sigma, hstep, htail, tau, bg0, a=1):
        """
        David Radford's HPGe peak shape function
        """
        # make sure the fractional amplitude parameters stay reasonable...
        if htail < 0 or htail > 1:
            return np.zeros_like(x)
        if hstep < 0 or hstep > 1:
            return np.zeros_like(x)

        bg_term = bg0  #+ x*bg1
        if np.any(bg_term < 0):
            return np.zeros_like(x)

        # compute the step and the low energy tail
        step = a * hstep * erfc((x - mu) / (sigma * np.sqrt(2)))
        le_tail = a * htail
        le_tail *= erfc((x - mu) / (sigma * np.sqrt(2)) + sigma / (tau * np.sqrt(2)))
        le_tail *= np.exp((x - mu) / tau)
        le_tail /= (2 * tau * np.exp(-(sigma / (np.sqrt(2) * tau))**2))

        # add up all the peak shape components
        return (1 - htail) * gauss(x, mu, sigma, a) + bg_term + step + le_tail

    hist, bins, var = pgh.get_hist(df['e_cal'], range=(495,525), dx=0.5)
    pgh.plot_hist(hist, bins, var=hist, label="data")
    pars, cov = pga.fit_hist(radford_peak, hist, bins, var=hist, guess=[511, 1.6, 0.003, 0.07, 2, 3, 600])
    #pgu.print_fit_results(pars, cov, radford_peak)
    pgu.plot_func(radford_peak, pars, label="chi2 fit", color='red')
    #x_vals = np.arange(490,540,0.5)
    #plt.plot(x_vals, radford_peak(x_vals, 511, 1.6, .003, 0.07, 5, 8, 500))

    FWHM = '%.2f' % Decimal(pars[1]*2)
    FWHM_uncertainty = '%.2f' % Decimal(np.sqrt(cov[1][1])*2)
    peak = '%.2f' % Decimal(pars[0])
    peak_uncertainty = '%.2f' % Decimal(np.sqrt(cov[0][0]))

    chi_2_element_list = []
    for i in range(len(hist)):
        chi_2_element = abs((radford_peak(bins[i], *pars) - hist[i])**2/radford_peak(bins[i], *pars))
        chi_2_element_list.append(chi_2_element)
    chi_2 = sum(chi_2_element_list)
    reduced_chi_2 = '%.2f' % Decimal(chi_2/len(hist))

    label_01 = '511 keV peak fit'
    label_02 = 'FWHM = '+str(FWHM)+r' $\pm$ '+str(FWHM_uncertainty)
    label_03 = 'Peak = '+str(peak)+r' $\pm$ '+str(peak_uncertainty)
    label_04 = r'Reduced $\chi^2 = $'+str(reduced_chi_2)
    colors = ['red','red','red','red']
    lines = [Line2D([0], [0], color=c, lw=2) for c in colors]
    labels = [label_01, label_02, label_03, label_04]

    plt.xlim(495,525)
    plt.ylim(0,plt.ylim()[1])
    plt.xlabel('Energy (keV)', ha='right', x=1.0)
    plt.ylabel('Counts', ha='right', y=1.0)

    plt.tight_layout()
    plt.legend(lines, labels, frameon=False, loc='upper right', fontsize='small')
    #plt.semilogy()
    plt.show()

def peak_239():

    df =  pd.read_hdf("Spectrum_203.hdf5", key="df")

    def gauss(x, mu, sigma, A=1):
        """
        define a gaussian distribution, w/ args: mu, sigma, area (optional).
        """
        return A * (1. / sigma / np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2. * sigma**2))

    def radford_peak(x, mu, sigma, hstep, htail, tau, bg0, a=1):
        """
        David Radford's HPGe peak shape function
        """
        # make sure the fractional amplitude parameters stay reasonable...
        if htail < 0 or htail > 1:
            return np.zeros_like(x)
        if hstep < 0 or hstep > 1:
            return np.zeros_like(x)

        bg_term = bg0  #+ x*bg1
        if np.any(bg_term < 0):
            return np.zeros_like(x)

        # compute the step and the low energy tail
        step = a * hstep * erfc((x - mu) / (sigma * np.sqrt(2)))
        le_tail = a * htail
        le_tail *= erfc((x - mu) / (sigma * np.sqrt(2)) + sigma / (tau * np.sqrt(2)))
        le_tail *= np.exp((x - mu) / tau)
        le_tail /= (2 * tau * np.exp(-(sigma / (np.sqrt(2) * tau))**2))

        # add up all the peak shape components
        return (1 - htail) * gauss(x, mu, sigma, a) + bg_term + step + le_tail

    hist, bins, var = pgh.get_hist(df['e_cal'], range=(220,260), dx=0.5)
    pgh.plot_hist(hist, bins, var=hist, label="data")
    pars, cov = pga.fit_hist(radford_peak, hist, bins, var=hist, guess=[238.6, 1.6, .003, 0.7, 300, 250, 600])
    #pgu.print_fit_results(pars, cov, radford_peak)
    pgu.plot_func(radford_peak, pars, label="chi2 fit", color='red')
    #x_vals = np.arange(220,260,0.5)
    #plt.plot(x_vals, radford_peak(x_vals, 238.6, 1.6, .003, 0.07, 300, 250, 600))

    FWHM = '%.2f' % Decimal(pars[1]*2)
    FWHM_uncertainty = '%.2f' % Decimal(np.sqrt(cov[1][1])*2)
    peak = '%.2f' % Decimal(pars[0])
    peak_uncertainty = '%.2f' % Decimal(np.sqrt(cov[0][0]))

    chi_2_element_list = []
    for i in range(len(hist)):
        chi_2_element = abs((radford_peak(bins[i], *pars) - hist[i])**2/radford_peak(bins[i], *pars))
        chi_2_element_list.append(chi_2_element)
    chi_2 = sum(chi_2_element_list)
    reduced_chi_2 = '%.2f' % Decimal(chi_2/len(hist))

    label_01 = '238.6 keV peak fit'
    label_02 = 'FWHM = '+str(FWHM)+r' $\pm$ '+str(FWHM_uncertainty)
    label_03 = 'Peak = '+str(peak)+r' $\pm$ '+str(peak_uncertainty)
    label_04 = r'Reduced $\chi^2 = $'+str(reduced_chi_2)
    colors = ['red','red','red','red']
    lines = [Line2D([0], [0], color=c, lw=2) for c in colors]
    labels = [label_01, label_02, label_03, label_04]

    plt.xlim(220,260)
    plt.ylim(0,plt.ylim()[1])
    plt.xlabel('Energy (keV)', ha='right', x=1.0)
    plt.ylabel('Counts', ha='right', y=1.0)

    plt.tight_layout()
    plt.legend(lines, labels, frameon=False, loc='upper right', fontsize='small')
    #plt.semilogy()
    plt.show()


if __name__ == '__main__':
        main()

