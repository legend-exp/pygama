import numpy as np
import pandas as pd
import os
import json
import glob

import pygama.math.peak_fitting as pgf
import pygama.math.histogram as pgh
import pygama.lgdo.lh5_store as lh5


def generate_cuts(data, parameters, verbose=False):
    """
    Finds double sided cut boundaries for a file for the parameters specified 
    
    Parameters
    ----------
    data : lh5 table or dictionary of arrays
                data to calculate cuts on
    parameters : dict
                 dictionary with the parameter to be cut and the number of sigmas to cut at
    """

    output_dict = {}
    for par in parameters.keys():
        num_sigmas = parameters[par]
        par_array = data[par]
        if not isinstance(par_array, np.ndarray):
            par_array = par_array.nda
        counts, start_bins, var = pgh.get_hist(par_array,10**5)
        max_idx = np.argmax(counts)
        mu = start_bins[max_idx]
        bin_range=1000
        
        if max_idx <bin_range:
            lower_bound_idx = 0
        else:
            lower_bound_idx = max_idx-bin_range
        lower_bound = start_bins[lower_bound_idx]
        
        if max_idx >len(start_bins)-bin_range:
            upper_bound_idx = -1
        else:
            upper_bound_idx = max_idx+bin_range
        
        upper_bound = start_bins[upper_bound_idx]
        
        try:
            counts, bins, var = pgh.get_hist(par_array,bins = 1000, range = (lower_bound, upper_bound))
            
            bin_centres = pgh.get_bin_centers(bins)
            
            fwhm = pgh.get_fwhm(counts, bins)[0]
            mean = float(bin_centres[np.argmax(counts)])
            std = fwhm/2.355
        except IndexError:
            bin_range=5000
        
            if max_idx <bin_range:
                lower_bound_idx = 0
            else:
                lower_bound_idx = max_idx-bin_range
            lower_bound = start_bins[lower_bound_idx]
            
            if max_idx >len(start_bins)-bin_range:
                upper_bound_idx = -1
            else:
                upper_bound_idx = max_idx+bin_range
            upper_bound = start_bins[upper_bound_idx]
            counts, bins, var = pgh.get_hist(par_array,bins = 1000, range = (lower_bound, upper_bound))
            
            bin_centres = pgh.get_bin_centers(bins)
            
            fwhm = pgh.get_fwhm(counts, bins)[0]
            mean = float(bin_centres[np.argmax(counts)])
            std = fwhm/2.355
        
        

        if isinstance(num_sigmas, (int, float)):
            num_sigmas_left = num_sigmas
            num_sigmas_right = num_sigmas
        elif isinstance(num_sigmas, dict):
            num_sigmas_left = num_sigmas["left"]
            num_sigmas_right = num_sigmas["right"]
        upper =float( (num_sigmas_right *std)+mean)
        lower = float((-num_sigmas_left *std)+mean)
        output_dict[par] ={'Mean Value': mean, 'Sigmas Cut': num_sigmas, 'Upper Boundary' : upper, 'Lower Boundary': lower}
    return output_dict

def get_cut_indexes(all_data, cut_dict, energy_param = 'trapEmax', verbose=False):

    """
    Returns a mask of the data, for a single file, that passes cuts based on dictionary of cuts 
    in form of cut boundaries above
    Parameters
    ----------
    File : dict or lh5_table
           dictionary of parameters + array such as load_nda or lh5 table of params
    Cut_dict : string
                Dictionary file with cuts
    """
    
    indexes = None
    keys = cut_dict.keys()
    for cut in keys:
        data = all_data[cut]
        if not isinstance(data, np.ndarray):
            data = data.nda
        
        if 'Energy Dep' in cut_dict[cut].keys():
            energy_midpoints, uppers, lowers = get_energy_dep(all_data, cut, 
                                                              n_sigmas= cut_dict[cut]['Energy Dep']['Cuts Specified'],
                                                               energy_param= energy_param, n_windows=10, verbose=verbose)
            pars = np.polynomial.polynomial.polyfit(energy_midpoints, uppers, deg=2)
            pars2 = np.polynomial.polynomial.polyfit(energy_midpoints, lowers, deg=2)
            upper_bounds = np.polynomial.polynomial.polyval(all_data['trapEmax'], pars)
            lower_bounds = np.polynomial.polynomial.polyval(all_data['trapEmax'], pars2)          
            idxs = (data<upper_bounds)& (data>lower_bounds)
        else:
            upper = cut_dict[cut]['Upper Boundary']
            lower = cut_dict[cut]['Lower Boundary']
            idxs = (data<upper) & (data>lower) 
        
        # Combine masks
        if indexes is not None:
            indexes = indexes & idxs
            
        else:
            indexes = idxs
        if verbose: print(cut, ' loaded')

    return indexes

