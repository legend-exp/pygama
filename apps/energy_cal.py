#!/usr/bin/env python3

import sys, os
import argparse
import numpy as np
import pandas as pd
import tinydb as db
import json
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
    
    par = argparse.ArgumentParser(description="pygama calibration suite")
    arg, st, sf = par.add_argument, "store_true", "store_false"
    arg("-f", nargs=1, help="filename and path ie. /path/to/file/lpgta_r1.lh5")
    arg("-h5p", nargs=1, help="path to hdf5 dataset ie. g034/raw")
    arg("-peakdet", action=st, help="run peakdet on raw spectrum for initial guesses")
    arg("-DB", nargs=1, help="json file with raw peak guesses and true energy")
    arg("-degree", nargs=1, help="What degree polynomial to calibrate to")
    arg("-write_db", action=st, help="store results in DB")
    
    args = vars(par.parse_args())
    
    #lpgta
    # "/Volumes/LaCie/Data/LPGTA/dsp/geds/LPGTA_r0018_20200302T184529Z_calib_geds_dsp.lh5"
    files = args['f'][0]
    groupname = args['h5p'][0]
    e_param = "trapE"
    
    #cage
    # file_location = "/Volumes/LaCie/Data/CAGE/LH5/dsp/icpc/"
    # file_list = ["cage_run8_cyc128_dsp.lh5", "cage_run8_cyc130_dsp.lh5",
    #                 "cage_run8_cyc129_dsp.lh5", "cage_run8_cyc131_dsp.lh5"]
                
    
    #HADES
    # file_location = "/Volumes/LaCie/Data/HADES/dsp/I02160A/"
    # file_list = ["hades_I02160A_r1_191021T162944_th_HS2_top_psa_dsp.lh5", 
    #             "hades_I02160A_r1_191021T163144_th_HS2_top_psa_dsp.lh5", 
    #             "hades_I02160A_r1_191021T163344_th_HS2_top_psa_dsp.lh5",
    #             "hades_I02160A_r1_191023T092533_th_HS2_lat_psa_dsp.lh5",
    #             "hades_I02160A_r1_191023T092733_th_HS2_lat_psa_dsp.lh5",
    #             "hades_I02160A_r1_191023T092933_th_HS2_lat_psa_dsp.lh5"]
    # 
    # groupname = "/raw"

    # 
    # files = []
    # for file in file_list:
    #     f = file_location + file
    #     files.append(f)
    
    # groupname = "/ORSIS3302DecoderForEnergy"
    # e_param = "trapE"
    
    
    energy = get_data(files, groupname, e_param)
    hE, xE, var = histo_data(energy, 0, np.max(energy), 1)
    if args["peakdet"]:
        find_peaks(hE, xE, var)
        exit()
    with open(args["DB"][0]) as f:
        pks_DB = json.load(f)
    par, perr, peaks = cal_input(hE, xE, var, energy, int(args["degree"][0]), pks_DB, write_db=args["write_db"])#, test=True)
    # resolution(par, energy, peaks, paramDB, 2)
    
    


def get_data(files, groupname, e_param='trapE'):
    """
     loop over file list, access energy array from LH5, concat arrays together
    return array
    """
    dsp = lh5.Store()
    energies = []
    
    if isinstance(files, list):
    
        for file in files:
            filename = os.path.expandvars(file)
            data = dsp.read_object(groupname, filename)
            energy = data[e_param].nda
            energies.extend(energy)
    else:
        filename = os.path.expandvars(files)
        data = dsp.read_object(groupname, filename)
        energy = data[e_param].nda
        energies.extend(energy)
        
        
    return np.asarray(energies)
        

def histo_data(array, elo, ehi, epb):
    """
    return histo array
    """
    
    hE, xE, var = ph.get_hist(array, range=[elo, ehi], dx=epb)


    return hE, xE, var


def find_peaks(hE, xE, var):
    
    """
    run peakdet routine (use a JSON config file to set thresholds)
    """
    
    maxes, mins = pu.peakdet(hE, 100, xE[1:])
    umaxes = np.array(sorted([x[0] for x in maxes]))
    print(f"{umaxes}")
    
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

def cal_input(hE, xE, var, e_array, degree, pks_DB, write_db=False, test=False):
    """
    -- mode of 'calibrate'
    -- access a JSON file wth expected peak locations for several peaks, compute a quadratic calibration
    """
    
    peak_table = pks_DB["peak_table"]
    #     '212Pb':238.6, '214Pb':351.9, 'beta+':511.0, '583':583.2, 
    #     '214Bi':609.3, '228Ac':911.2, '228Ac':969.0,
    #     '40K':1460.8, 'DEP':1592, '214Bi':1764.5, 'SEP':2104, '208Tl':2614.5
    # }
    #cage
    # expected_peaks = ['212Pb', 'beta+', '214Bi', '208Tl']
    # raw_peaks_guess = np.asarray([406, 872, 3009, 4461])
    
    #lpgta
    # expected_peaks = ['212Pb', '583', 'DEP', 'SEP', '208Tl']
    # raw_peaks_guess = np.asarray([1894, 3861, 9521, 12404, 15426])
    expected_peaks = pks_DB["expected_peaks"]
    raw_peaks_guess = np.asarray(pks_DB["raw_peak_guesses"])
    
    #hades
    # expected_peaks = ['212Pb', '583', 'DEP', 'SEP', '208Tl']
    # raw_peaks_guess = np.asarray([3124, 8394, 23710, 31430, 39172])
    
    
    raw_peaks = np.array([])
    raw_error = np.array([])
    for pk in raw_peaks_guess:
        
        h, e_range, var1 = ph.get_hist(e_array, range=[pk-50, pk+50], dx=1)
        e_range = e_range[1:]
        h_sub = h - np.min(h)
        i_max = np.argmax(h)
        h_max = h[i_max]
        hs_max = h_sub[i_max]
        upr_half = e_range[np.where((e_range > e_range[i_max]) & (h_sub <= hs_max/2))][0]
        bot_half = e_range[np.where((e_range < e_range[i_max]) & (h_sub <= hs_max/2))][-1]
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

    weights = np.diag(1 / error**2)
    
    raw_peaks_matrix = np.zeros((len(raw_peaks),degree+1))
    if len(raw_peaks) < degree + 1:
        print(f"cannot calibrate to degree {degree} polynomial if there are less than {degree + 1} raw peaks")
        exit()
    
    for i, pk in enumerate(raw_peaks):
        
        temp_degree = degree
        row = np.array([])
        
        while temp_degree >= 0:
            row = np.append(row, pk**temp_degree)
            temp_degree -= 1
        
        raw_peaks_matrix[i] += row

    xTWX = np.dot(np.dot(raw_peaks_matrix.T, weights), raw_peaks_matrix)
    xTWY = np.dot(np.dot(raw_peaks_matrix.T, weights), true_peaks)
    xTWX_inv = np.linalg.inv(xTWX) 
    par = np.dot(xTWX_inv, xTWY) 
    perr = np.sqrt(np.diag(xTWX_inv))
    
    print(f"{par}")
    print(f"{perr}")
    
    ecal = np.zeros((1, len(e_array)))
    
    cal_peaks = np.zeros(len(raw_peaks))
    
    temp_degree = degree
    for i in range(len(par)):
        ecal += e_array**temp_degree * par[i]
        cal_peaks += raw_peaks**temp_degree * par[i]
        temp_degree -= 1
    print(cal_peaks)
    print(true_peaks)
        
    residuals = true_peaks - cal_peaks


    
    hcal, xcal, var = ph.get_hist(ecal, range=[0, 3500], dx=1)
    xcal = xcal[1:]
    
    initial_guesses = init_guesses(hcal, xcal, cal_peaks)
    
    
    
    cmap = plt.cm.get_cmap('jet', len(true_peaks) + 1)
    
    for i in range(len(true_peaks)):
        plt.vlines(true_peaks[i], 0, 30000, color=cmap(i), linestyle="--", lw=1, label=true_peaks[i])
    
    plt.semilogy(xcal, hcal, ls='steps', lw=1, c='r', label=f"a={par[0]:.4} b={par[1]:.4} c={par[2]:.4} ")
    plt.xlabel("Energy", ha='right', x=1)
    plt.ylabel("Counts", ha='right', y=1)
    plt.title(f"Cal hist degree {degree}")
    plt.legend()
    plt.tight_layout()
    if test == True:
        plt.show()
    plt.savefig('e_hist_cal.png')
    plt.clf()
    
    plt.errorbar(true_peaks, residuals, yerr=raw_error, ls='none', capsize=5, marker=".", ms=10)
    plt.hlines(0, 0, 3000, color= 'r')
    plt.title(f"WLS degree {degree} residuals")
    plt.xlabel("TrueE", ha='right', x=1)
    plt.ylabel("Residuals", ha='right', y=1)
    if test == True:
        plt.show()
    plt.savefig('e_residuals.png')
    plt.clf()

    if write_db == True:
        paramDB = db.TinyDB('cal_pars.json')
        paramDB.insert({'params':par.tolist()})
        paramDB.insert({'perr':perr.tolist()})
    
    resolution(par, e_array, true_peaks, initial_guesses, degree)
    
    return par, perr, cal_peaks
    
    
def write_output():
    """
    -- get cal constants, covariance matrix, results of peak search
    -- write to file
    """
def init_guesses(e_cal_hist, xE, peaks, test=False):
    
    initial_guesses = []
    
    for pk in peaks:
        
        h = e_cal_hist[np.where((pk-25 < xE) & (xE < pk+25))]
        e_range = xE[np.where((pk-25 < xE) & (xE < pk+25))]
        h_sub = h - np.min(h)
        i_max = np.argmax(h)
        h_max = h[i_max]
        hs_max = h_sub[i_max]
        upr_half = e_range[np.where((e_range > e_range[i_max]) & (h_sub <= hs_max/2))][0]
        bot_half = e_range[np.where((e_range < e_range[i_max]) & (h_sub <= hs_max/2))][-1]
        fwhm = upr_half - bot_half
        sig = fwhm / 2.355
        p0 = [e_range[i_max], h_max, sig, 0, 0]
        
        par, pcov = curve_fit(simple_gauss, e_range, h, p0=p0, sigma = np.sqrt(h), absolute_sigma=True)
        perr = np.sqrt(np.diag(pcov))
        print(par, perr)
        if test==True:
            plt.plot(e_range, h, ls='steps', lw=1, c='r')
            plt.plot(e_range, simple_gauss(e_range, par[0], par[1], par[2], par[3], par[4]))
            plt.errorbar(e_range, h, yerr=np.sqrt(h), ls='none')
            plt.show()
        
        initial_guesses.append(par.tolist())
    return initial_guesses
    
    
def resolution(par, e_array, peaks, initial_guesses, degree):
    
    params = initial_guesses
    
    ecal = np.zeros((1, len(e_array)))
    for i in range(len(par)):
        ecal += e_array**degree * par[i]
        degree -= 1
    
    resolution = np.array([])
    res_error = np.array([])
    
    for i, pk in enumerate(peaks):
        
        h, e_range, var = ph.get_hist(ecal, range=[pk-(1.2*params[i][2]*2.355), pk+(1.2*params[i][2]*2.355)], dx=.5)
        
        i_max = np.argmax(h)
        h_max = h[i_max]
        amp = h_max * params[i][2] * 2.355
        # hstep = 0.01 # fraction that the step contributes
        # htail = 0.1
        # tau = 10
        # bg0 = params[i][4] + params[i][3]*e_range[0]
        # x0 = [params[i][0], params[i][2], hstep, htail, tau, bg0, amp]
        # radford_par, radford_cov = pf.fit_hist(pf.radford_peak, h, e_range, var=np.sqrt(h), guess=x0)
        # radford_err = np.sqrt(np.diag(radford_cov))
        # fit_func = pf.radford_peak
        
        p0 = [e_range[i_max], h_max, params[i][2], e_range[2]]
        par1, pcov = curve_fit(gauss, e_range[1:], h, p0=p0)#, sigma = np.sqrt(h), absolute_sigma=True)
        perr = np.sqrt(np.diag(pcov))
        
        
        
        # plt.plot(e_range[1:], h, ls='steps', lw=1, c='r')
        # plt.plot(e_range[1:], gauss(e_range[1:], *par1))
        # plt.show()
        
        resolution = np.append(resolution, par1[2]*2.355)
        res_error = np.append(res_error, perr[2]*2.355)
    # exit()
    
    plt.errorbar(peaks, resolution, yerr=res_error, ls='none', capsize=5, marker=".", ms=10)
    plt.title("Resolution vs E")
    plt.xlabel("keV")
    plt.ylabel("FWHM")
    # plt.show()
    plt.savefig('e_resolution.png')
    
    
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
