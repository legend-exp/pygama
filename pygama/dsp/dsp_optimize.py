import numpy as np
from .build_processing_chain import build_processing_chain
from collections import namedtuple


def run_one_dsp(tb_data, dsp_config, functional=None, verbosity=0):
    """
    run one iteration of DSP on tb_data 

    Optionally returns a value for optimization

    Parameters:
    -----------
    tb_data : lh5 Table
        An input table of lh5 data. Typically a selection is made prior to
        sending tb_data to this function: optimization typically doesn't have to
        run over all data
    dsp_config : dict
        Specifies the DSP to be performed for this iteration (see
        build_processing_chain()) and the list of output variables to appear in
        the output table
    functional : function or None (optional)
        When given the output lh5 table of this DSP iteration, the functional
        must return a scalar real value upon which the optimization will be
        based. Should accept verbosity as a second argument
    verbosity : int (optional)
        verbosity for the processing chain and functional calls

    Returns:
    --------
    functional_value : float
        If functional is not None, returns functional value for the DSP
        iteration
    tb_out : lh5 Table
        If functional is None, returns the output lh5 table for the DSP
        iteration
    """
    
    pc, tb_out = build_processing_chain(tb_data, dsp_config, verbosity=verbosity)
    pc.execute()
    if functional is not None: return functional(tb_out, verbosity)
    else: return tb_out



ParGridDimension = namedtuple('ParGridDimension', 'name i_arg value_strs')

class ParGrid():
    """ Parameter Grid class

    Each ParGrid entry corresponds to a dsp parameter to be varied.
    The ntuples must follow the pattern: 
    ( name, i_arg, value_strs ) : ( str, int, array of str )
    where name is the name of the dsp routine in dsp_config whose  be
    optimized, i_arg is the index of the argument to be varied, value_strs is
    the array of strings to set the argument to
    """
    def __init__(self):
        self.dims = []

    def add_dimension(self, name, i_arg, value_strs):
        self.dims.append( ParGridDimension(name, i_arg, value_strs) )

    def get_n_dimensions(self):
        return len(self.dims)

    def get_n_points_of_dim(self, i):
        return len(self.dims[i].value_strs)

    def get_shape(self):
        shape = ()
        for i in range(self.get_n_dimensions()):
            shape += (self.get_n_points_of_dim(i),)
        return shape

    def get_n_grid_points(self):
        return np.prod(self.get_shape())

    def get_par_meshgrid(self, copy=False, sparse=False):
        """ return a meshgrid of parameter values

        Always uses Matrix indexing (natural for par grid) so that
        mg[i1][i2][...] corresponds to index order in self.dims

        Note copy is False by default as opposed to numpy default of True
        """     
        axes = []
        for i in range(self.get_n_dimensions()):
            axes.append(self.dims[i].values_strs)
        return np.meshgrid(*axes, copy, sparse, indexing='ij')

    def get_zero_indices(self):
        return np.zeros(self.get_n_dimensions(), dtype=np.uint32)

    def iterate_indices(self, indices):
        """ iterate given indices [i1, i2, ...] by one.

        For easier iteration. The convention here is arbitrary, but its the
        order the arrays would be traversed in a series of nested for loops in
        the order appearin in dims (first dimension is first for loop, etc):

        Return False when the grid runs out of indices. Otherwise returns True.
        """
        for iD in reversed(range(self.get_n_dimensions())):
            indices[iD] += 1
            if indices[iD] < self.get_n_points_of_dim(iD): return True
            indices[iD] = 0
        return False

    def get_data(self, i_dim, i_par):
        name = self.dims[i_dim].name
        i_arg = self.dims[i_dim].i_arg
        value_str = self.dims[i_dim].value_strs[i_par]
        return name, i_arg, value_str

    def print_data(self, indices):
        print(f"Grid point at indices {indices}:")
        for i_dim, i_par in enumerate(indices):
            name, i_arg, value_str = self.get_data(i_dim, i_par)
            print(f"{name}[{i_arg}] = {value_str}")

    def set_dsp_pars(self, dsp_config, indices):
        for i_dim, i_par in enumerate(indices):
            name, i_arg, value_str = self.get_data(i_dim, i_par)
            dsp_config['processors'][name]['args'][i_arg] = value_str


def run_grid(tb_data, dsp_config, grid, functional, verbosity=0):
    """Extract a table of optimization values for a grid of DSP parameters 

    The grid argument defines a list of parameters and values over which to run
    the DSP defined in dsp_config on tb_data. At each point, a functional is
    used to extract a scalar value. 

    Returns a N-dimensional ndarray of scalar values, where the array axes are
    in the order they appear in grid.

    Parameters:
    -----------
    tb_data : lh5 Table
        An input table of lh5 data. Typically a selection is made prior to
        sending tb_data to this function: optimization typically doesn't have to
        run over all data
    dsp_config : dict
        Specifies the DSP to be performed (see build_processing_chain()) and the
        list of output variables to appear in the output table for each grid point
    grid : ParGrid
        See ParGrid class for format
    functional : function 
        When given the output lh5 table of this DSP iteration, the functional
        must return a scalar real value upon which the optimization will be
        based. Should accept verbosity as a second keyword argument
    verbosity : int (optional)
        verbosity for the processing chain and functional calls

    Returns:
    --------
    grid_values : ndarray of floats
        An N-dimensional numpy ndarray whose Mth axis corresponds to the Mth row
        of the grid argument
    """

    grid_values = np.ndarray(shape=grid.get_shape())
    iii = grid.get_zero_indices()
    if verbosity > 0: print("starting grid calculations...")
    while True:    
        grid.set_dsp_pars(dsp_config, iii)
        if verbosity > 0: grid.print_data(iii)
        grid_values[tuple(iii)] = run_one_dsp(tb_data, dsp_config, functional, verbosity)
        if verbosity > 0: print("value:", grid_values[tuple(iii)])
        if not grid.iterate_indices(iii): break
    return grid_values
        


'''
#-----------------------------------

#!/usr/bin/env python3
import os
import re
import time
import json
import argparse
import pandas as pd
import numpy as np
from pprint import pprint
from datetime import datetime
import itertools
from collections import OrderedDict
from scipy.optimize import curve_fit

import tinydb as db
from tinydb.storages import MemoryStorage

import matplotlib
if os.environ.get('HOSTNAME'): # cenpa-rocks
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('../clint.mpl')
from matplotlib.colors import LogNorm

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from tqdm import tqdm
    tqdm.pandas() # suppress annoying FutureWarning

from pygama import DataGroup
from pygama.dsp.units import *
from pygama.dsp.ProcessingChain import ProcessingChain
import pygama.io.lh5 as lh5
import pygama.analysis.histograms as pgh
import pygama.analysis.peak_fitting as pgf


def main():
    doc="""
    === optimizer.py ====================================================

    dsp optimization app, works with DataGroup

    === C. Wiseman (UW) =============================================
    """
    rthf = argparse.RawTextHelpFormatter
    par = argparse.ArgumentParser(description=doc, formatter_class=rthf)
    arg, st, sf = par.add_argument, 'store_true', 'store_false'
    
    # primary operations
    arg('-q', '--query', nargs=1, type=str,
        help="select group to analyze: -q 'cycle==1' ")
    arg('-e', '--energy', action=st, help='optimize energy trapezoid')
    arg('-d', '--dcr', action=st, help='optimize DCR parameter')

    args = par.parse_args()
    
    # -- setup -- 
    
    # load main DataGroup, select files to analyze
    dg = DataGroup('cage.json', load=True)
    if args.query:
        que = args.query[0]
        dg.file_keys.query(que, inplace=True)
    else:
        dg.file_keys = dg.file_keys[-1:]
 
    view_cols = ['run','cycle','daq_file','runtype','startTime','threshold']
    print(dg.file_keys[view_cols].to_string())
    # print(f'Found {len(dg.file_keys)} files.')
    
    # -- run routines -- 
    
    # TODO : we could split this code into "spectrum" (peak width) optimizations, 
    # and "waveform" optimizations, where the FOM is a waveform, not a peak.
    # so like optimize_spec.py and optimize_wfs.py
    
    optimize_trap(dg)
    show_trap_results()
    
    # optimize_dcr(dg) 
    # show_dcr_results(dg)
    # check_wfs(dg)
    
    
def optimize_trap(dg):
    """
    Generate a file with grid points to search, and events from the target peak.  
    Then run DSP a bunch of times on the small table, and fit the peak w/ the
    peakshape function.  
    NOTE: run table-to-table DSP (no file I/O)
    """
    f_peak = './temp_peak.lh5' # lh5
    f_results = './temp_results.h5' # pandas
    grp_data, grp_grid = '/optimize_data', '/optimize_grid'
    
    # epar, elo, ehi, epb = 'energy', 0, 1e7, 10000 # full range
    epar, elo, ehi, epb = 'energy', 3.88e6, 3.92e6, 500 # K40 peak
    
    show_movie = True
    write_output = True
    n_rows = None # default None
    
    with open('opt_trap.json') as f:
        dsp_config = json.load(f, object_pairs_hook=OrderedDict)
    
    # files to consider.  fixme: right now only works with one file
    sto = lh5.Store()
    lh5_dir = os.path.expandvars(dg.config['lh5_dir'])
    raw_list = lh5_dir + dg.file_keys['raw_path'] + '/' + dg.file_keys['raw_file']
    f_raw = raw_list.values[0] 
    tb_raw = 'ORSIS3302DecoderForEnergy/raw/'

    # quick check of the energy range
    # ene_raw = sto.read_object(tb_raw+'/'+epar, f_raw).nda
    # hist, bins, var = pgh.get_hist(ene_raw, range=(elo, ehi), dx=epb)
    # plt.plot(bins[1:], hist, ds='steps')
    # plt.show()
    # exit()
    
    # set grid parameters
    # TODO: jason's suggestions, knowing the expected shape of the noise curve
    # e_rises = np.linspace(-1, 0, sqrt(sqrt(3))
    # e_rises # make another list which is 10^pwr of this list
    # np.linspace(log_tau_min, log_tau_max) # try this too
    e_rises = np.arange(1, 12, 1)
    e_flats = np.arange(1, 6, 1)
    # rc_consts = np.arange(54, 154, 10) # changing this here messes up DCR
    
    # -- create the grid search file the first time -- 
    # NOTE: this makes a linear grid, and is editable by the arrays above.
    # jason also proposed a more active gradient-descent style search
    # like with Brent's method. (https://en.wikipedia.org/wiki/Brent%27s_method)
    
    if True:
    # if not os.path.exists(f_peak):
        print('Recreating grid search file')
        
        # create the grid file
        # NOTE: save it as an lh5 Table just as an example of writing/reading one
        lists = [e_rises, e_flats]#, rc_consts]
        prod = list(itertools.product(*lists)) # clint <3 stackoverflow
        df_grid = pd.DataFrame(prod, columns=['rise', 'flat'])#,'rc']) 
        lh5_grid = {}
        for i, dfcol in df_grid.iteritems():
            lh5_grid[dfcol.name] = lh5.Array(dfcol.values)
        tb_grid = lh5.Table(col_dict=lh5_grid)
        sto.write_object(tb_grid, grp_grid, f_peak)
            
        # filter events by onboard energy
        ene_raw = sto.read_object(tb_raw+'/'+epar, f_raw).nda
        # hist, bins, var = pgh.get_hist(ene_raw, range=(elo, ehi), dx=epb)
        # plt.plot(bins[1:], hist, ds='steps')
        # plt.show()
        if n_rows is not None:
            ene_raw = ene_raw[:n_rows]
        idx = np.where((ene_raw > elo) & (ene_raw < ehi))

        # create a filtered table with correct waveform and attrs
        # TODO: move this into a function in lh5.py which takes idx as an input
        tb_data, wf_tb_data = lh5.Table(), lh5.Table()

        # read non-wf cols (lh5 Arrays)
        data_raw = sto.read_object(tb_raw, f_raw, n_rows=n_rows)
        for col in data_raw.keys():
            if col=='waveform': continue
            newcol = lh5.Array(data_raw[col].nda[idx], attrs=data_raw[col].attrs)
            tb_data.add_field(col, newcol)
        
        # handle waveform column (lh5 Table)
        data_wfs = sto.read_object(tb_raw+'/waveform', f_raw, n_rows=n_rows)
        for col in data_wfs.keys():
            attrs = data_wfs[col].attrs
            if isinstance(data_wfs[col], lh5.ArrayOfEqualSizedArrays):
                # idk why i can't put the filtered array into the constructor
                aoesa = lh5.ArrayOfEqualSizedArrays(attrs=attrs, dims=[1,1])
                aoesa.nda = data_wfs[col].nda[idx]
                newcol = aoesa
            else:
                newcol = lh5.Array(data_wfs[col].nda[idx], attrs=attrs)
            wf_tb_data.add_field(col, newcol)
        tb_data.add_field('waveform', wf_tb_data)
        tb_data.attrs = data_raw.attrs
        sto.write_object(tb_data, grp_data, f_peak)

    else:
        print('Loading peak file. groups:', sto.ls(f_peak))
        tb_grid = sto.read_object(grp_grid, f_peak)
        tb_data = sto.read_object(grp_data, f_peak) # filtered file
        # tb_data = sto.read_object(tb_raw, f_raw) # orig file
        df_grid = tb_grid.get_dataframe()
        
    # check shape of input table
    print('input table attributes:')
    for key in tb_data.keys():
        obj = tb_data[key]
        if isinstance(obj, lh5.Table):
            for key2 in obj.keys():
                obj2 = obj[key2]
                print('  ', key, key2, obj2.nda.shape, obj2.attrs)
        else:
            print('  ', key, obj.nda.shape, obj.attrs)

    # clear new colums if they exist
    new_cols = ['e_fit', 'fwhm_fit', 'rchisq', 'xF_err', 'fwhm_ovr_mean']
    for col in new_cols:
        if col in df_grid.columns:
            df_grid.drop(col, axis=1, inplace=True)

    t_start = time.time()
    def run_dsp(dfrow):
        """
        run dsp on the test file, editing the processor list
        alternate idea: generate a long list of processors with different names
        """
        # adjust dsp config dictionary
        rise, flat = dfrow
        # dsp_config['processors']['wf_pz']['defaults']['db.pz.tau'] = f'{tau}*us'
        dsp_config['processors']['wf_trap']['args'][1] = f'{rise}*us'
        dsp_config['processors']['wf_trap']['args'][2] = f'{flat}*us'
        # pprint(dsp_config)
        
        # run dsp
        pc, tb_out = build_processing_chain(tb_data, dsp_config, verbosity=0)
        pc.execute()
        
        # analyze peak
        e_peak = 1460.
        etype = 'trapEmax'
        elo, ehi, epb = 4000, 4500, 3 # the peak moves around a bunch
        energy = tb_out[etype].nda
        
        # get histogram
        hE, bins, vE = pgh.get_hist(energy, range=(elo, ehi), dx=epb)
        xE = bins[1:]
        
        # should I center the max at 1460?

        # simple numerical width
        i_max = np.argmax(hE)
        h_max = hE[i_max]
        upr_half = xE[(xE > xE[i_max]) & (hE <= h_max/2)][0]
        bot_half = xE[(xE < xE[i_max]) & (hE >= h_max/2)][0]
        fwhm = upr_half - bot_half
        sig = fwhm / 2.355
        
        # fit to gaussian: amp, mu, sig, bkg
        fit_func = pgf.gauss_bkg
        amp = h_max * fwhm
        bg0 = np.mean(hE[:20])
        x0 = [amp, xE[i_max], sig, bg0]
        xF, xF_cov = pgf.fit_hist(fit_func, hE, bins, var=vE, guess=x0)

        # collect results
        e_fit = xF[0]
        xF_err = np.sqrt(np.diag(xF_cov))
        e_err = xF
        fwhm_fit = xF[1] * 2.355 * 1460. / e_fit
        
        fwhm_err = xF_err[2] * 2.355 * 1460. / e_fit
        
        chisq = []
        for i, h in enumerate(hE):
            model = fit_func(xE[i], *xF)
            diff = (model - h)**2 / model
            chisq.append(abs(diff))
        rchisq = sum(np.array(chisq) / len(hE))
        fwhm_ovr_mean = fwhm_fit / e_fit

        if show_movie:
            
            plt.plot(xE, hE, ds='steps', c='b', lw=2, label=f'{etype} {rise}--{flat}')

            # peak shape
            plt.plot(xE, fit_func(xE, *x0), '-', c='orange', alpha=0.5,
                     label='init. guess')
            plt.plot(xE, fit_func(xE, *xF), '-r', alpha=0.8, label='peakshape fit')
            plt.plot(np.nan, np.nan, '-w', label=f'mu={e_fit:.1f}, fwhm={fwhm_fit:.2f}')

            plt.xlabel(etype, ha='right', x=1)
            plt.ylabel('Counts', ha='right', y=1)
            plt.legend(loc=2)

            # show a little movie
            plt.show(block=False)
            plt.pause(0.01)
            plt.cla()

        # return results
        return pd.Series({'e_fit':e_fit, 'fwhm_fit':fwhm_fit, 'rchisq':rchisq,
                          'fwhm_err':xF_err[0], 'fwhm_ovr_mean': fwhm_ovr_mean})
    
    # df_grid=df_grid[:10]
    df_tmp = df_grid.progress_apply(run_dsp, axis=1)
    df_grid[new_cols] = df_tmp
    # print(df_grid)
    
    if show_movie:
        plt.close()
    
    print('elapsed:', time.time() - t_start)
    if write_output:
        df_grid.to_hdf(f_results, key=grp_grid)
        print(f"Wrote output file: {f_results}")

'''
