#!/usr/bin/env python3

import sys, os
import argparse
import numpy as np
import pandas as pd
import tinydb as db
import matplotlib.pyplot as plt
import itertools as it
from scipy.stats import mode
from pprint import pprint

import pygama.utils as pu
import pygama.analysis.histograms as ph
import pygama.analysis.peak_fitting as pf
from pygama.io import lh5



def main():
    
    #cage
    file_location = "/global/cfs/cdirs/legend/data/cage/LH5/dsp/icpc/"
    file_list = ["cage_run8_cyc128_dsp.lh5", "cage_run8_cyc130_dsp.lh5",
                    "cage_run8_cyc129_dsp.lh5", "cage_run8_cyc131_dsp.lh5"]
                
    
    #HADES
    # file_location = "/global/cfs/cdirs/legend/users/wisecg/HADES"
    # file_list = ["hades_I02160A_r1_191021T162944_th_HS2_top_psa_dsp.lh5", 
    #             "hades_I02160A_r1_191021T163144_th_HS2_top_psa_dsp.lh5", 
    #             "hades_I02160A_r1_191021T163344_th_HS2_top_psa_dsp.lh5"]
    
    files = []
    for file in file_list:
        f = file_location + file
        files.append(f)

    # groupname = "/raw"
    groupname = "/ORSIS3302DecoderForEnergy"
    e_param = "trapE"
    
    
    energy = get_data(files, groupname, e_param)
    histo_data(energy, 0, np.max(energy), 10)
    


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
    # plt.savefig("un_cal.png")
    return hE, xE, var

def find_peaks(histogram):
    
    """
    run peakdet routine (use a JSON config file to set thresholds)
    """
    
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

def cal_input():
    """
    -- mode of 'calibrate'
    -- access a JSON file wth expected peak locations for several peaks, compute a quadratic calibration
    """
def write_output():
    """
    -- get cal constants, covariance matrix, results of peak search
    -- write to file
    """



if __name__=="__main__":
    main()
