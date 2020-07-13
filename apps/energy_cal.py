#!/usr/bin/env python3

import sys, os
import argparse
import numpy as np
import pandas as pd
import tinydb as db
import matplotlib.pyplot as plt
import itertools as it
from time import process_time
from scipy.stats import mode
from scipy.optimize import curve_fit
from scipy.optimize import minimize 
from pprint import pprint

import pygama.utils as pu
import pygama.analysis.histograms as ph
import pygama.analysis.peak_fitting as pf
from pygama.io import lh5



def main():
    
    paramDB = db.TinyDB('params.json')
    
    #cage
    file_location = "/Volumes/LaCie/Data/CAGE/LH5/dsp/icpc/"
    file_list = ["cage_run8_cyc128_dsp.lh5", "cage_run8_cyc130_dsp.lh5",
                    "cage_run8_cyc129_dsp.lh5", "cage_run8_cyc131_dsp.lh5"]
                    
    """
    Datagroup not working because paths to files are specific to cori right now, just need to change 
    the json files to get this to work
    """
    # dg = DataGroup('CAGE.json')
    # dg.load_df('CAGE_fileDB.h5')
    # que = 'run==8'
    # dg.file_keys.query(que, inplace=True)
                
    
    #HADES
    # file_location = "/global/cfs/cdirs/legend/users/wisecg/HADES"
    # file_list = ["hades_I02160A_r1_191021T162944_th_HS2_top_psa_dsp.lh5", 
    #             "hades_I02160A_r1_191021T163144_th_HS2_top_psa_dsp.lh5", 
    #             "hades_I02160A_r1_191021T163344_th_HS2_top_psa_dsp.lh5"]
    
    files = []
    for file in file_list:
        f = file_location + file
        files.append(f)

    groupname = "/ORSIS3302DecoderForEnergy"
    e_param = "trapE"
    
    
    energy = get_data(files, groupname, e_param)
    hE, xE, var = histo_data(energy, 0, np.max(energy), 1)
    find_peaks(hE, xE, var)
    # par, perr, peaks = cal_input(hE, xE, var, energy)
    # resolution(par, energy, peaks, paramDB)
    
    


def get_data(files, groupname, e_param):
    """
     loop over file list, access energy array from LH5, concat arrays together
    return array
    """
    dsp = lh5.Store()
    energies = []
    
    for file in files:
        filename = os.path.expandvars(file)
        data = dsp.read_object(groupname, filename)
        energy = data[e_param].nda
        energies.extend(energy)
        
    return np.asarray(energies)
        

def histo_data(array, elo, ehi, epb):
    """
    return histo array
    """
    
    hE, xE, var = ph.get_hist(array, range=[elo, ehi], dx=epb)
    
    # plt.semilogy(xE[1:], hE, ls='steps', lw=1, c='r')
    # 
    # plt.xlabel("Energy (uncal.)", ha='right', x=1)
    # plt.ylabel("Counts", ha='right', y=1)
    # plt.show()
    return hE, xE, var


def find_peaks(hE, xE, var):
    
    """
    run peakdet routine (use a JSON config file to set thresholds)
    """
    
    maxes, mins = pu.peakdet(hE, 150, xE[1:])
    umaxes = np.array(sorted([x[0] for x in maxes]))
    print(umaxes)
    
    for peak in umaxes:
        plt.axvline(peak, linestyle="--", lw=1)
    
    plt.semilogy(xE[1:], hE, ls='steps', lw=1, c='r')
    
    plt.xlabel("Energy (uncal.)", ha='right', x=1)
    plt.ylabel("Counts", ha='right', y=1)
    plt.show()
    
    
def calibrate(histogram, peak_list, test_peaks, mode):
    """
    call functions for each mode to get calibration constants
    run weighted least squares analysis
    return cal constants and covariance matrix,
    """

def ratio_match():
    """
    mode of 'calibrate'
    find linear calibration by
    """

def save_template():
    """
    after any calibration mode, save a calibrated histogram for this channel
    """
def template_match(histogram, reference_histogram):
    """
    -- mode of 'calibrate'
    -- access a file with a reference histogram for this detector, and shift/scale histogram to minimize chi2
    """

def cal_input(hE, xE, var, e_array, test=False):
    """
    -- mode of 'calibrate'
    -- access a JSON file wth expected peak locations for several peaks, compute a quadratic calibration
    """
    
    peak_table = {
        '212Pb':238.6, '214Pb':351.9, 'beta+':511.0, '208Tl':583.2, 
        '214Bi':609.3, '228Ac':911.2, '228Ac':969.0,
        '40K':1460.8, '214Bi':1764.5, '208Tl':2614.5
    }
    expected_peaks = ['212Pb', 'beta+', '40K', '214Bi', '208Tl']
    
    raw_peaks_guess = np.asarray([406, 872, 2490, 3009, 4461])
    
    raw_peaks = np.array([])
    raw_error = np.array([])
    for pk in raw_peaks_guess:
        
        h, e_range, var1 = ph.get_hist(e_array, range=[pk-15, pk+15], dx=1)
        e_range = e_range[1:]
        h_sub = h - np.min(h)
        i_max = np.argmax(h)
        h_max = h[i_max]
        hs_max = h_sub[i_max]
        upr_half = e_range[(e_range > e_range[i_max]) & (h_sub <= hs_max/2)][0]
        bot_half = e_range[(e_range < e_range[i_max]) & (h_sub >= hs_max/2)][0]
        fwhm = upr_half - bot_half
        sig = fwhm / 2.355
        p0 = [e_range[i_max], h_max, sig, 0, 0]
        
        par, pcov = curve_fit(simple_gauss, e_range, h, p0=p0, sigma = np.sqrt(h), absolute_sigma=True)
        perr = np.sqrt(np.diag(pcov))
        if test == True:
            print(par)
            print(perr)
            plt.plot(e_range, h, ls='steps', lw=1, c='r')
            plt.plot(e_range, simple_gauss(e_range, par[0], par[1], par[2], par[3], par[4]))
            plt.errorbar(e_range, h, yerr=np.sqrt(h), ls='none')
            plt.show()
            
        raw_peaks = np.append(raw_peaks, par[0])
        raw_error = np.append(raw_error, perr[0])
    

    true_peaks = np.array([peak_table[pk] for pk in expected_peaks])
    error = raw_error / raw_peaks * true_peaks
    cov = np.diag(error**2)

    # weights = 1 / error**2
    weights = np.diag(1 / error**2)
    
    
    #####1st order cal constants
    
    # start = process_time()
    # par, cov = np.polyfit(raw_peaks, true_peaks, 1, w=weights, cov=True)
    # end = process_time()
    # poly1_error = np.sqrt(np.diag(cov))
    # ecal = e_array * par[0] + par[1]
    

    # start = process_time()
    # par, pcov = curve_fit(line, raw_peaks, true_peaks, sigma=error)
    # end = process_time()
    # print(par)
    # perr = np.sqrt(np.diag(pcov))
    # print(perr)
    # ecal = e_array * par[0]
    
    # def wls(pars, x_vals, data_points):
    # 
    #     wls = weights * (data_points - line(x_vals, pars[0], pars[1]))**2
    #     wls = np.asarray(wls)
    #     return np.sum(wls)
    # 
    # x0 = [.5, .1]
    # start = process_time()
    # min = minimize(wls, x0, args=(true_peaks, raw_peaks))
    # end = process_time()
    # par = min['x']
    # ecal = (e_array) /  par[0]
    
    # raw_peaks_matrix = np.array([[1,raw_peaks[0]], [1,raw_peaks[1]], [1, raw_peaks[2]], [1,raw_peaks[3]]])
    # xTWX = np.dot(np.dot(raw_peaks_matrix.T, weights), raw_peaks_matrix)
    # xTWY = np.dot(np.dot(raw_peaks_matrix.T, weights), true_peaks)
    # 
    # xTWX_inv = np.linalg.inv(xTWX) 
    # par = np.dot(xTWX_inv, xTWY) 
    # ecal = e_array * par[1]
    
    
    ####2nd order
    
    # start = process_time()
    # par, cov = np.polyfit(raw_peaks, true_peaks, 2, w=weights, cov=True)
    # end = process_time()
    # perr = np.sqrt(np.diag(cov))
    # ecal = par[0]*e_array**2 + par[1]*e_array + par[2]
    # print(f"{par}")
    # print(f"{perr}")

    
    # start = process_time()
    # par, pcov = curve_fit(quadratic, raw_peaks, true_peaks, sigma=cov, absolute_sigma=True)
    # end = process_time()
    # print(par)
    # perr = np.sqrt(np.diag(pcov))
    # print(perr)
    # ecal = par[0]*e_array**2 + par[1]*e_array + par[2]
    
    # def wls(pars, x_vals, data_points):
    # 
    #     wls = weights * (x_vals - quadratic(data_points, pars[0], pars[1], pars[2]))**2
    #     wls = np.asarray(wls)
    #     return np.sum(wls)
    # 
    # x0 = [.1, .5,3]
    # start = process_time()
    # min = minimize(wls, x0, args=(true_peaks, raw_peaks))
    # end = process_time()
    # par = min['x']
    # ecal = par[0]*e_array**2 + par[1]*e_array
    
    raw_peaks_matrix = np.array([[raw_peaks[0]**2,raw_peaks[0], 1], 
                                [raw_peaks[1]**2, raw_peaks[1], 1], 
                                [raw_peaks[2]**2, raw_peaks[2], 1], 
                                [raw_peaks[3]**2, raw_peaks[3], 1],
                                [raw_peaks[4]**2, raw_peaks[4], 1]])
    xTWX = np.dot(np.dot(raw_peaks_matrix.T, weights), raw_peaks_matrix)
    xTWY = np.dot(np.dot(raw_peaks_matrix.T, weights), true_peaks)
    
    xTWX_inv = np.linalg.inv(xTWX) 
    par = np.dot(xTWX_inv, xTWY) 
    perr = np.sqrt(np.diag(xTWX_inv))
    print(f"{par}")
    print(f"{perr}")
    
    ecal = e_array**2 * par[0] + e_array * par[1] + par[2]
    
    
    # true_peaks1 = true_peaks / (par[0]*raw_peaks**2 + par[1]*raw_peaks + par[2]) - 1
    residuals = true_peaks - (par[0]*raw_peaks**2 + par[1]*raw_peaks + par[2])
    # raw_peaks1 = raw_peaks /  (true_peaks*par[0]+par[1])-1
    # error = np.sqrt(variance) / (true_peaks*par[0]+par[1])
    # print(raw_peaks1)
    # print(f"{residuals}")
    # print(f"{rerror}")
    
    hcal, xcal, var = ph.get_hist(ecal, range=[0, 3500], dx=.5)
    xcal = xcal[1:]
    
    initial_guesses = init_guesses(hcal, xcal, true_peaks)
    
    
    
    cmap = plt.cm.get_cmap('jet', len(true_peaks) + 1)
    if test == True:
        # for i in range(len(true_peaks)):
        #     plt.vlines(true_peaks[i], 0, 11000, color=cmap(i), linestyle="--", lw=1, label=true_peaks[i])
        
        # plt.semilogy(xcal, hcal, ls='steps', lw=1, c='r', label=f"a={par[0]:.4} b={par[1]:.4} c={par[2]:.4}")
        # plt.scatter(true_peaks, residuals, s=10)#, label="a={} b={} c={}".format(par[0], par[1], par[2]))
        # plt.errorbar(true_peaks, residuals, yerr=raw_error, ls='none')
        # plt.plot(raw_peaks, quadratic(raw_peaks, par[0], par[1], par[2]), color='red', label="a={} b={} c={}".format(par[0], par[1], par[2]))
        # plt.hlines(0, 0, 3000, color= 'r')
        # plt.xlabel("Energy", ha='right', x=1)
        # plt.ylabel("Counts", ha='right', y=1)
        # plt.xlabel("TrueE", ha='right', x=1)
        # plt.ylabel("Residuals", ha='right', y=1)
        # plt.title("Cal hist quad polyfit no constant")
        # plt.title("Cal hist quad curvefit")
        # plt.title("Cal hist quad WLS")
        # plt.title("Poly fit linear normalized")
        # plt.title("WLS quad residuals")
        # plt.legend()
        # plt.tight_layout()
        # plt.show()
    
    # paramDB = db.TinyDB('params.json')
    # paramDB.insert({'par0':initial_guesses})
    # exit()
    
    
    return par, perr, true_peaks
    
    
def write_output():
    """
    -- get cal constants, covariance matrix, results of peak search
    -- write to file
    """
def init_guesses(e_cal_hist, xE, peaks):
    
    initial_guesses = []
    
    for pk in peaks:
        
        h = e_cal_hist[np.where((pk-25 < xE) & (xE < pk+25))]
        e_range = xE[np.where((pk-25 < xE) & (xE < pk+25))]
        h_sub = h - np.min(h)
        i_max = np.argmax(h)
        h_max = h[i_max]
        hs_max = h_sub[i_max]
        upr_half = e_range[(e_range > e_range[i_max]) & (h_sub <= hs_max/2)][0]
        bot_half = e_range[(e_range < e_range[i_max]) & (h_sub >= hs_max/2)][0]
        fwhm = upr_half - bot_half
        sig = fwhm / 2.355
        p0 = [e_range[i_max], h_max, sig, 0, 0]
        
        par, pcov = curve_fit(simple_gauss, e_range, h, p0=p0, sigma = np.sqrt(h), absolute_sigma=True)
        perr = np.sqrt(np.diag(pcov))
        # print(par, perr)
        # plt.plot(e_range, h, ls='steps', lw=1, c='r')
        # plt.plot(e_range, simple_gauss(e_range, par[0], par[1], par[2], par[3], par[4]))
        # plt.errorbar(e_range, h, yerr=np.sqrt(h), ls='none')
        # plt.show()
        
        initial_guesses.append(par.tolist())
    return initial_guesses
    
    
def resolution(par, e_array, peaks, paramDB):
    
    params = paramDB.all()[0]['par0']
    ecal = e_array**2 * par[0] + e_array * par[1] + par[2]
    
    resolution = np.array([])
    res_error = np.array([])
    
    for i, pk in enumerate(peaks):
        
        h, e_range, var = ph.get_hist(ecal, range=[pk-(params[i][2]+7), pk+(params[i][2]+7)], dx=.5)
        
        i_max = np.argmax(h)
        h_max = h[i_max]
        amp = h_max * params[i][2] * 2.355
        hstep = 0.01 # fraction that the step contributes
        htail = 0.1
        tau = 10
        bg0 = params[i][4] + params[i][3]*e_range[2]
        x0 = [params[i][0], params[i][2], hstep, htail, tau, bg0, amp]
        radford_par, radford_cov = pf.fit_hist(pf.radford_peak, h, e_range, var=np.sqrt(h), guess=x0)
        radford_err = np.sqrt(np.diag(radford_cov))
        fit_func = pf.radford_peak
        
        # plt.plot(e_range[1:], h, ls='steps', lw=1, c='r')
        # plt.plot(e_range[1:], fit_func(e_range[1:], *radford_par))
        # plt.show()
        
        resolution = np.append(resolution, radford_par[1])
        res_error = np.append(res_error, radford_err[1])
    
    plt.clf()
    plt.errorbar(peaks, resolution, yerr=res_error, ls='none', capsize=5, marker=".", ms=10)
    plt.title("Resolution vs E")
    plt.xlabel("keV")
    plt.ylabel("FWHM")
    plt.show()
    
    
def line(x, a, b):
    return a*x + b

def quadratic(x, a, b, c):
    return a*x**2 + b*x + c

def gauss(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params) - 1, 3):
        x0 = params[i]
        a = params[i + 1]
        sigma = params[i + 2]
        y += a * np.exp(-(x - x0)**2 / (2 * sigma**2))
    y = y + params[-1]
    return y
    
def simple_gauss(x, x0, a, sigma, b, const):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + b*x + const


if __name__=="__main__":
    main()
