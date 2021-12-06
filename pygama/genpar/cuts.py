import numpy as np
import pandas as pd
import os
import json
from pygama.analysis import histograms as pgh
import pygama.analysis.peak_fitting as pgf
import pygama.lh5 as lh5
import glob

def generate_cuts(data, parameters, verbose=False):
    """
    Finds double sided cut boundaries for the dat using the parameters specified 
    
    Parameters
    ----------
    data : lh5 table or dictionary of arrays
                data to calculate cuts on
    parameters : dict
                 dictionary with the parameter to be cut and the number of sigmas to cut at
    """

    output_dict = {}
    for par in list(parameters):
        num_sigmas = parameters[par]
        par_array = data[par]
        if not isinstance(par_array, np.ndarray):
            par_array = par_array.nda
        counts, bins, var = pgh.get_hist(par_array,10**5)
        guess_pars = pgh.get_gaussian_guess(counts, bins)
        lower_bound = guess_pars[0]-10*guess_pars[1]
        upper_bound = guess_pars[0]+10*guess_pars[1]
        counts, bins, var = pgh.get_hist(par_array,bins = 1000, range = (lower_bound, upper_bound))
        pars, cov = pgf.fit_hist(pgf.gauss, counts, bins, var, guess = guess_pars, bounds = get_bounds(*guess_pars, verbose=verbose))
        mean,std,area = pars
        if isinstance(num_sigmas, (int, float)):
            num_sigmas_left = num_sigmas_right = num_sigmas
        elif isinstance(num_sigmas, dict):
            num_sigmas_left, num_sigmas_right = num_sigmas["left"], num_sigmas["right"]
        upper =float(mean+(num_sigmas_right *std))
        lower = float(mean-(num_sigmas_left *std))
        output_dict[par] ={'Mean Value': mean, 'Sigmas Cut': num_sigmas, 'Upper Boundary' : upper, 'Lower Boundary': lower}
    return output_dict

def get_bounds(mean, sigma, area, verbose=False):
    """
    Calculates bounds for gaussian peak fitting
    """
    if mean >0:
        mean_lims = [0.75*mean, 1.25*mean]
    else:
        if verbose:print("mean<0")
        mean_lims = [1.25*mean, 0.75*mean]
    sigma_lims = [0,10*sigma]
    area_lims = [0,2*area]
    return list(zip(mean_lims, sigma_lims, area_lims))

def get_cut_boundaries(file_path,  lh5_group, parameters = {'bl_mean':4,'bl_std':4, 'pz_std':4}, 
                       energy_param= 'trapTmax', cut_file=None, overwrite=False, verbose=False):
    
    """
    Finds double sided cut boundaries for a file for the parameters specified 
    
    Parameters
    ----------
    file_path : str
                path to data file
    cut_file : str
                path to json file of cuts will save cuts in the form:
                {detector-source-run : {parameter : {Mean_value : mean_value, Sigmas cut : sigmas, 
                                                     Upper Boundary : value, Lower Boundary : value}}
    lh5_group : str
                lh5_path (e.g. 'raw')
    parameters : dict
                 dictionary with the parameter to be cut and the number of sigmas to cut at
    overwrite : bool
                True to overwrite existing cuts in json file
    """


    if os.path.isfile(cut_file) == True:
        cut_dict = json.load(open(cut_file,'r'))

    else:
        cut_dict = {}
    
    base=os.path.basename(file_path)
    file_name = os.path.splitext(base)[0]
    datatype, detector, measurement, run, timestamp = file_name.split('-')
    det_run = f'{datatype}-{detector}-{measurement}-{run}'

    if det_run in cut_dict and overwrite == False:
        print('Cut parameters already calculated and overwrite is false')
        return
    all_params = list(parameters)
    all_params.append(energy_param)
    all_pars_array = lh5.load_nda(file_path, all_params,  lh5_group, verbose=verbose)
    output_dict = generate_cuts(all_pars_array, parameters)
    energy_dep_dict = check_energy_dep(data, cut_parameters, energy_param)
    output_dict.update(energy_dep_dict)
    cut_dict[det_run] = output_dict
    if cut_file:
        with open(cut_file,'w') as fp:
            json.dump(cut_dict,fp, indent=4)
    return

def compare_dicts(dict1, dict2):
    """
    Used to compare 2 dictionaries for evaluating energy dependence of cuts 
    """
    parameters = list(dict1)
    output_dict = {}
    for par in parameters:
        upper1 = dict1[par]['Upper Boundary']
        upper2 = dict2[par]['Upper Boundary']
        lower1 = dict1[par]['Lower Boundary']
        lower2 = dict2[par]['Lower Boundary']
        if lower2<0:
            if upper1>1.1*upper2 or upper1<0.9*upper2 or lower1>0.9*lower2 or lower1<1.1*lower2:
                output_dict[par] = 'Energy Dep'
        else:
            if upper1>1.1*upper2 or upper1<0.9*upper2 or lower1>1.1*lower2 or lower1<0.9*lower2:
                output_dict[par] = 'Energy Dep'
    return output_dict

def check_energy_dep(data, parameters, energy_param, verbose=False):
    """
    Checks energy dependence of cut parameters by calculating cut values in 2 windows
    """
    energy = data[energy_param]
    max_val = np.nanpercentile(energy,99) #Removes extreme events
    half_max = max_val/2
    low_window = (energy<half_max)
    high_window = (energy>half_max)
    low_wind_data ={}
    high_wind_data ={}
    for key in list(parameters):
        low_wind_data[key] = data[key][low_window]
        high_wind_data[key] = data[key][high_window]
    low_wind_cut_dict = generate_cuts(low_wind_data, parameters, verbose=verbose)
    high_wind_cut_dict = generate_cuts(high_wind_data, parameters, verbose=verbose)
    out_dict = compare_dicts(low_wind_cut_dict, high_wind_cut_dict)
    for pars in list(out_dict):
        out_dict[pars] = {'Energy Dep':{'Cuts Specified':parameters[pars]}} 
    return out_dict


def get_energy_dep(data, parameter, n_sigmas=4, energy_param='trapEmax', n_windows=10, verbose=False):
    
    energy = data[energy_param]
    n_events = len(energy)
    if n_events < 40000:
        print("Due to energy dependence use more events to get accurate cuts")
    max_val = np.percentile(energy,99)
    energy_win = max_val/n_windows
    windows = energy_midpoints = []
    for n in range(n_windows):
        energy_midpoints.append(((n+1)*energy_win+(n)*energy_win)/2)
        window = (energy<(n+1)*energy_win) * (energy>(n)*energy_win)
        windows.append(window)
    uppers = []
    lowers = []
    final_energies=[]
    par_dict = {parameter:n_sigmas}
    for i,window in enumerate(windows):
        win_data = data[parameter][window]
        if len(win_data)>0.02*n_events and len(win_data)>1000:
            out_dict = generate_cuts({parameter:win_data}, par_dict, verbose=verbose)
            uppers.append(out_dict[parameter]['Upper Boundary'])
            lowers.append(out_dict[parameter]['Lower Boundary'])
            final_energies.append(energy_midpoints[i])
    return final_energies, uppers, lowers

def get_cut_indexes(all_data, cut_dict, energy_param = 'trapEmax', verbose=False):

    """
    Returns a mask of the data, for a single file, that passes cuts based on dictionary 
    containing cut boundaries as above
    Parameters
    ----------
    File : dict or lh5_table
           dictionary of parameters + array such as load_nda or lh5 table of params
    Cut_dict : string
                Dictionary file with cuts
    """
    
    indexes = None
    for cut in list(cut_dict):
        data = all_data[cut]
        if not isinstance(data, np.ndarray):
            data = data.nda
        
        if 'Energy Dep' in list(cut_dict[cut]):
            energy_midpoints, uppers, lowers = get_energy_dep(all_data, cut, 
                                                              n_sigmas= cut_dict[cut]['Energy Dep']['Cuts Specified'],
                                                               energy_param= energy_param, n_windows=10, verbose=verbose)
            upper_pars = np.polynomial.polynomial.polyfit(energy_midpoints, uppers, deg=2)
            lower_pars = np.polynomial.polynomial.polyfit(energy_midpoints, lowers, deg=2)
            upper_bounds = np.polynomial.polynomial.polyval(all_data['trapEmax'], upper_pars)
            lower_bounds = np.polynomial.polynomial.polyval(all_data['trapEmax'], lower_pars)          
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

def load_cut_mask(files, lh5_group, cut_file=None,  cut_parameters= {'bl_mean':4,'bl_std':4, 'pz_std':4}, 
                       energy_param = 'trapTmax',verbose=False):

    """
    This function returns a mask of the waveforms passing cuts.
    You can specify these cut parameters with a dictionary. If you supply a cut file json it will load and save the cuts to this 
    file in the form of get_cut_boundaries
    Parameters
    ----------
    files : List
            list of file paths
    lh5_group : string
                lh5 file path e.g. 'raw/'
    parameters : str, list
                parameters to load
    
    cut_file : string
                Path to json dictionary file with cuts, if provided cuts will be saved/loaded from here.
    
    cut_parameters : dict
                Dictionary in the form {cut:no_sigmas}, can be double sided {cut:{"left":no_sigmas,"right":no_sigmas}}
                if cut_file provided and cuts found there will overrule cut_parameters, if none found will save these to cut_file
    
    energy_param :  str
                Parameter used to determine the energy dependence of the cuts and to correct for them if found
    
    """

    sto = lh5.Store()

    if isinstance(files, str): files = [files]
    # Expand wildcards
    files = [f for f_wc in files for f in sorted(glob.glob(os.path.expandvars(f_wc)))]

    base=os.path.basename(files[0])
    file_name = os.path.splitext(base)[0]
    datatype, detector, measurement, run, timestamp = file_name.split('-')
    run_1st = f'{datatype}-{detector}-{measurement}-{run}'

    all_params = list(cut_parameters)
    all_params.append(energy_param)

    if cut_file is None:
        data = lh5.load_nda(files[0], all_params, lh5_group, verbose=verbose)
        cut_dict = generate_cuts(data, cut_parameters)
        energy_dep_dict = check_energy_dep(data, cut_parameters, energy_param)
        cut_dict.update(energy_dep_dict)
        print("Generated Cut Dictionary")
        if verbose: print(cut_dict)
        idxs = []
        for file in files:
            base=os.path.basename(file)
            file_name = os.path.splitext(base)[0]
            datatype, detector, measurement, run, timestamp = file_name.split('-')
            det_run = f'{datatype}-{detector}-{measurement}-{run}'
            if det_run != run_1st:
                run_1st = det_run
                print ("Files not all in same run")
                data = lh5.load_nda(file, all_params, lh5_group, verbose=verbose)
                cut_dict = generate_cuts(data, cut_parameters)
                energy_dep_dict = check_energy_dep(data, cut_parameters, energy_param)
                cut_dict.update(energy_dep_dict)
                print(cut_dict)
            par_data = lh5.load_nda(file, all_params, lh5_group, verbose=verbose)
            idx = get_cut_indexes(par_data, cut_dict, energy_param, verbose)
            idxs.append(idx)

    else:
        try:
            with open(cut_file,'r') as f:
                full_cut_dict = json.load(f)
            cut_dict = full_cut_dict[run_1st]
            if verbose: print('Loaded Cut Dictionary')
        except:
            print("Cuts haven't been calculated yet, getting cut boundaries")
            get_cut_boundaries(files[0], cut_file, lh5_group, energy_param = energy_param, parameters=cut_parameters)
            with open(cut_file,'r') as f:
                full_cut_dict = json.load(f)
                cut_dict = full_cut_dict[run_1st]

        idxs = []
        for file in files:
            if verbose: print(f"loading data for {file}")

            base=os.path.basename(files[0])
            file_name = os.path.splitext(base)[0]
            datatype, detector, measurement, run, timestamp = file_name.split('-')
            det_run = f'{datatype}-{detector}-{measurement}-{run}'
            if det_run != run_1st:
                print ("Files not all in same run")
                run_1st = det_run
                try:
                    cut_dict = full_cut_dict[run_1st]
                except KeyError:
                    print("Cuts haven't been calculated yet, getting cut boundaries")
                    get_cut_boundaries(files[0], cut_file, lh5_group, energy_param = energy_param, parameters=cut_parameters)
                    with open(cut_file,'r') as f:
                        full_cut_dict = json.load(f)
                        cut_dict = full_cut_dict[run_1st]
            keys = list(cut_dict)
            keys.append(energy_param)
            par_data = lh5.load_nda(file, keys, lh5_group, verbose=verbose)
            idx = get_cut_indexes(par_data, cut_dict, energy_param, verbose)
            idxs.append(idx)
    
    mask = np.concatenate(idxs)
    print(f"{len(np.where(mask)[0])/ len(mask) *100:1.2f}% passed cuts")
    return mask

def load_nda_with_cuts(files, parameters, lh5_group, cut_file=None,  cut_parameters= {'bl_mean':4,'bl_std':4, 'pz_std':4}, 
                       energy_param = 'trapEmax',verbose=False):

    """
    This function loads data after applying cuts, by default it will simply load the data according to the default cut parameters. 
    You can specify these cut parameters with a dictionary. If you supply a cut file json it will load and save the cuts to this 
    file in the form of get_cut_boundaries
    Parameters
    ----------
    files : List
            list of file paths

    parameters : str, list
                parameters to load
    
    lh5_group : string
                lh5 file path e.g. 'raw/'
    
    cut_file : string
                Path to json dictionary file with cuts, if provided cuts will be saved/loaded from here.
    
    cut_parameters : dict
                Dictionary in the form {cut:no_sigmas}, can be double sided {cut:{"left":no_sigmas,"right":no_sigmas}}
                if cut_file provided and cuts found there will overrule cut_parameters, if none found will save these to cut_file
    
    energy_param :  str
                Parameter used to determine the energy dependence of the cuts and to correct for them if found
    
    """

    sto = lh5.Store()

    if isinstance(files, str): files = [files]
    if isinstance(parameters, str): parameters = [parameters]
    # Expand wildcards
    files = [f for f_wc in files for f in sorted(glob.glob(os.path.expandvars(f_wc)))]

    mask = load_cut_mask(files, lh5_group, cut_file=cut_file, 
                        cut_parameters=cut_parameters, energy_param=energy_param, verbose=verbose)
    #Concat dicts together and return
    par_data = lh5.load_nda(files, parameters, lh5_group, verbose=verbose)
    failed_cuts={}
    passed_cuts={}
    for par in par_data:
        passed_cuts[par] = par_data[par][mask]
        failed_cuts[par] = par_data[par][~mask]
    return passed_cuts, failed_cuts

def load_df_with_cuts(files, lh5_group, cut_file=None, cut_parameters= {'bl_mean':4,'bl_std':4, 'pz_std':4}, 
                      energy_param = 'trapEmax',verbose=False):

    """
    This function loads all data after applying cuts, by default it will simply load the data according to the default cut parameters. 
    You can specify these cut parameters with a dictionary. If you supply a cut file json it will load and save the cuts to this 
    file in the form of get_cut_boundaries
    Parameters
    ----------
    files : List
            list of file paths
    
    lh5_group : string
                lh5 file path e.g. 'raw/'
    
    cut_file : string
                Path to json dictionary file with cuts, if provided cuts will be saved/loaded from here.
    
    cut_parameters :
                Dictionary in the form {cut:no_sigmas}, can be double sided {cut:{"left":no_sigmas,"right":no_sigmas}}
                if cut_file provided and cuts found there will overrule cut_parameters, if none found will save these to cut_file
    
    """

    sto = lh5.Store()
    if isinstance(files, str): files = [files]
    # Expand wildcards
    files = [f for f_wc in files for f in sorted(glob.glob(os.path.expandvars(f_wc)))]

    mask = load_cut_mask(files, lh5_group, cut_file=cut_file, 
                        cut_parameters=cut_parameters, energy_param=energy_param, verbose=verbose)
    
    tb = sto.read_object(lh5_group, files)[0]
    data = lh5.Table.get_dataframe(tb)
        
    cut_data = data.iloc[mask]
    failed_cuts = data.iloc[~mask]

    cut_data.reset_index(inplace= True)
    failed_cuts.reset_index(inplace= True)
    return cut_data, failed_cuts