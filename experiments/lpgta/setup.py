# import os module
import os

# turn off file locking
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# import python modules
import argparse
import collections
import copy
import glob
import h5py
import importlib
import json
import logging
import math
import matplotlib.backends.backend_pdf as pdf
import matplotlib.colors
import matplotlib.pyplot as plt
import multiprocessing as mp
import numba
import numpy as np
import scipy.odr as odr
import scipy.optimize as optimize
import sklearn.decomposition as decomposition
import sys
import warnings

# set up the argument parser
__rthf__ = argparse.RawTextHelpFormatter
__pars__ = argparse.ArgumentParser(formatter_class=__rthf__)

# include the arguments
__pars__.add_argument('-n', action='store_true'    , help='create a new database file')
__pars__.add_argument('-d', action='store_true'    , help='run in debug mode'         )
__pars__.add_argument('-o', type=str, default=''   , help='set the output directory'  )
__pars__.add_argument('-s', type=str, default=''   , help='set the analysis stage'    )
__pars__.add_argument('-f', type=str, default='cal', help='set the processing label'  )
__pars__.add_argument('-p', type=int, default=1    , help='set the number of poles'   )
__pars__.add_argument('-c', type=int, default=1    , help='set the number of cores'   )
__pars__.add_argument('-i', type=int, nargs='+'    , help='set the indices to process')
__pars__.add_argument('-g', type=str, nargs='+'    , help='set the geds to process'   )

# parse the arguments
__args__ = __pars__.parse_args()

# interpret the arguments
replace_db = __args__.n
debug_mode = __args__.d
output_dir = __args__.o
proc_stage = __args__.s
pole_count = __args__.p
proc_label = __args__.f
proc_count = __args__.c if __args__.c < mp.cpu_count() else mp.cpu_count()
proc_index = __args__.i
proc_chans = __args__.g

# exit if the provided arguments are invalid
if (pole_count == 2 and proc_label == 'cal') or\
   (pole_count not in [1, 2]               ) or\
   (proc_label not in ['cal', 'opt', 'dsp']):
    sys.exit('The provided arguments are invalid')
    
# silence warnings if requested
if not debug_mode:
    warnings.filterwarnings('ignore')

# setup the logging as requested
importlib.reload(logging)
prefix = '%(asctime)s: ' if debug_mode else ''
logging.basicConfig(format=f'{prefix}%(message)s', level=logging.INFO, datefmt='%H:%M:%S')
    
# the database file
__template__ = 'template.json'
__database__ = 'database.json'
    
# check if creating a new database
if replace_db:
    # exit if the template does not exist
    if not os.path.isfile(__template__):
        sys.exit('The template configuration was not found')

    # read in the file
    with open(__template__, 'r') as f:
        db = json.load(f)
        
    # the raw files to process
    files_raw = []

    # include the files
    for f in db['files_raw']:
        files_raw.extend(glob.glob(os.path.expandvars(f)))

    # sort the files
    files_raw = np.sort(files_raw).tolist()
    
    # update the database
    db['files_raw'] = files_raw
else:
    # read in the file
    with open(__database__, 'r') as f:
        db = json.load(f)
        
    # the raw files to process
    files_raw = db['files_raw']

# exit if there are no files to process
if len(files_raw) == 0:
    sys.exit('Found no files to process')

# update the system path
if db['pyg_dir'] != '':
    sys.path = [os.getcwd() + '/' + db['pyg_dir']] + sys.path

# import pygama modules
import pygama.dsp.build_processing_chain as bpc
import pygama.dsp.dsp_optimize as dspo
import pygama.dsp.processors as proc
import pygama.dsp.ProcessingChain as ProcessingChain
import pygama.dsp.units as units
import pygama.lh5 as lh5

# initialize the output directories
fig_dir = '.' if db['fig_dir'] == '' else db['fig_dir']
idx_dir = '.' if db['idx_dir'] == '' else db['idx_dir']
cal_dir = '.' if db['cal_dir'] == '' else db['cal_dir']
opt_dir = '.' if db['opt_dir'] == '' else db['opt_dir']
dsp_dir = '.' if db['dsp_dir'] == '' else db['dsp_dir']
ana_dir = '.' if db['ana_dir'] == '' else db['ana_dir']

# update the output directories
fig_dir = output_dir + '/' if output_dir != '' else fig_dir + '/'
idx_dir = output_dir + '/' if output_dir != '' else idx_dir + '/'
cal_dir = output_dir + '/' if output_dir != '' else cal_dir + '/'
opt_dir = output_dir + '/' if output_dir != '' else opt_dir + '/'
dsp_dir = output_dir + '/' if output_dir != '' else dsp_dir + '/'
ana_dir = output_dir + '/' if output_dir != '' else ana_dir + '/' + os.path.basename(sys.argv[0])[:-3] + '/'

# create the output directories if needed
if not os.path.exists(fig_dir): os.makedirs(fig_dir)
if not os.path.exists(idx_dir): os.makedirs(idx_dir)
if not os.path.exists(cal_dir): os.makedirs(cal_dir)
if not os.path.exists(opt_dir): os.makedirs(opt_dir)
if not os.path.exists(dsp_dir): os.makedirs(dsp_dir)

# the analysis labels
__labels__ = [
    'crosstalk',
    'noise',
    'pz_average',
    'pz_energy',
    'pz_preliminary',
    'pz_time',
    'resolution',
    'scatter']

# loop over the analysis labels
for label in __labels__:
    # the output directory
    out_dir = db['ana_dir'] + label
    
    # check if the output directory is missing
    if not os.path.exists(out_dir):
        # create the output directory
        os.makedirs(out_dir)

# the processed files
files_cal = [cal_dir + os.path.basename(f).replace('raw', 'cal') for f in files_raw]
files_opt = [opt_dir + os.path.basename(f).replace('raw', 'opt') for f in files_raw]
files_dsp = [dsp_dir + os.path.basename(f).replace('raw', 'dsp') for f in files_raw]
files_ana = [ana_dir + os.path.basename(f).replace('raw', 'ana') for f in files_raw]

# the files to process
if proc_label == 'cal': files_arg = files_cal
if proc_label == 'opt': files_arg = files_opt
if proc_label == 'dsp': files_arg = files_dsp

# check that the key is present
if 'channels' in db:
    # check if channels were specified
    if proc_chans:
        # the channels to process
        channels = proc_chans
    else:
        # the channels to process
        channels = list(db['channels'].keys())
        channels = np.sort(channels)
    
    # the number of cores
    if proc_count < 0:
        proc_count = min(len(channels), mp.cpu_count())

# read in the processors
with open('processors.json') as f:
    processors = json.load(f, object_pairs_hook=collections.OrderedDict)

# the lock to synchronize the processes
lock = mp.Lock()

# create a store object
store = lh5.Store()

# the rise and flat times in microseconds
rise = np.linspace(5, 26, int((26 - 5) / 1.0) + 1)
flat = np.linspace(0,  6, int(( 6 - 0) / 0.5) + 1)

# check that the key is present
if 'tick_width' in db and\
   'tick_count' in db:
    # the index bound
    index_hi = int(db['tick_count'] / 2)

    # calculate the combinations
    trap = []
    pick = []
    for r in rise:
        for f in flat:
            __r_ct__ = int(r / db['tick_width'])
            __f_ct__ = int(f / db['tick_width'])
            __p_ct__ = __r_ct__ + __f_ct__ - int(0.5 / db['tick_width'])
            if index_hi + __p_ct__ < db['tick_count']:
                trap.append([__r_ct__, __f_ct__])
                pick.append(f'tp_00+{__p_ct__}' )

# the model functions
def sqrt_func(x, a, b   ): return a * np.sqrt(x + b)
def cons_func(x, a      ): return a + 0 * x
def linr_func(x, a, b   ): return a + b * x
def expl_func(x, a, b, c): return a + b * np.exp(-x / c)
def gaus_func(x, a, b, c): return a * np.exp(-np.power(x - b, 2) / (2 * np.power(c, 2)))

# update the database on disk
def update_database(db):
    try:
        # try writing the temporary file
        with open('.' + __database__, 'w') as f:
            json.dump(db, f, indent=4)
            
        # try reading the temporary file
        with open('.' + __database__, 'r') as f:
            db = json.load(f)
    except:
        # delete the temporary file if it exists
        if os.path.isfile('.' + __database__):
            os.remove('.' + __database__)
    
        # exit if the update failed
        sys.exit('Updating the database was unsuccessful')

    # update the database
    os.replace('.' + __database__, __database__)
    
# write the index mask
def wr_mask(size, idx, label):
    # initialize the index mask
    mask = np.zeros(size, dtype='bool')
    
    # update the mask
    mask[idx] = True
    
    # write the mask to a binary file
    mask.tofile(idx_dir + label + '.bin')
    
# read the index mask
def rd_mask(label):
    return np.fromfile(idx_dir + label + '.bin', dtype='bool')
    
# find the mask for the specified file and channel
def find_mask(label, i, channel):
    # select the events
    beg = int(np.sum(db['channels'][channel]['event_count'][:i]))
    end = int(       db['channels'][channel]['event_count'][ i] ) + beg

    # return the mask
    return rd_mask(f'{channel}_{label}')[beg:end]

# calculate the bin to populate using the example:
# https://numba.pydata.org/numba-examples/examples/density_estimation/histogram/results.html
@numba.jit(nopython=True)
def compute_bin(x, bins):
    # the number of bins
    n = bins.shape[0] - 1
    
    # the range of bins
    b_min = bins[ 0]
    b_max = bins[-1]

    # check for special case to mirror numpy for the last bin
    if x == b_max:
        # return the last bin
        return n - 1 

    # the bin
    b = int(n * (x - b_min) / (b_max - b_min))

    # return the bin if valid
    if b < 0 or b >= n:
        return None
    else:
        return b

# histogram the data with numba using the example:
# https://numba.pydata.org/numba-examples/examples/density_estimation/histogram/results.html
@numba.jit(nopython=True)
def numba_histogram(data, bins):
    # the histogram
    hist = np.zeros((len(bins) - 1,), dtype=np.intp)

    # loop over the data
    for x in data.flat:
        # the bin
        b = compute_bin(x, bins)
        
        # update the histogram if valid
        if b is not None:
            hist[int(b)] += 1

    # return the histogram
    return hist
        
# calculate the peak
@numba.jit(nopython=True)
def find_peak(data, bins):
    # histogram the data
    n = numba_histogram(data, bins)
    
    # find the bin width
    bin_width = bins[1] - bins[0]
    
    # find the largest bin
    i = np.argmax(n)
    
    # calculate the peak
    xpeak = bins[i] + (bin_width / 2)
    if i > 0 and i < len(n) - 1 and n[i-1] != n[i+1]:
        xpeak += bin_width * (n[i-1] - n[i+1]) / (2 * (n[i-1] - 2 * n[i] + n[i+1]))
    
    # calculate the peak
    ypeak = n[i]
    if i > 0 and i < len(n) - 1 and n[i-1] != n[i+1]:
        ypeak -= np.power(n[i-1] - n[i+1], 2) / (8 * (n[i-1] - 2 * n[i] + n[i+1]))
    
    # return the peak
    return xpeak, ypeak

# calculate the bin width
def find_bin_width(data):
    # calculate the percentiles
    q1 = np.percentile(data, 25, interpolation='midpoint')
    q3 = np.percentile(data, 75, interpolation='midpoint')
    
    # return the bin width
    return 2 * (q3 - q1) / np.cbrt(len(data))

# calculate the bins
def find_bins(data, min_bin_width=None):
    # find the bin width
    bin_width = find_bin_width(data)
    
    # check the bound if provided
    if min_bin_width and\
       min_bin_width > bin_width:
        # set to the bound
        bin_width = min_bin_width
    
    # return the bins
    return np.arange(min(data), max(data), bin_width)

# the figure of merit to be calculated at each grid point
def figure_of_merit(eftp, corr, bins):
    # do a principle-component analysis for a first guess
    pca = decomposition.PCA(n_components=2)
    pca.fit(np.vstack((eftp - np.average(eftp), corr - np.average(corr))).T)
    de, dc = pca.components_[np.argmax(pca.explained_variance_)]
    
    # the figures of merit
    scales = np.linspace(0, 2, 201) * (de / dc)
    values = []

    # loop over the correction scales
    for scale in scales:
        # find the peak
        _, height = find_peak(eftp - corr * scale, bins)

        # include the figure of merit
        values.append(height)

    # return the result
    return scales, values

# plot the matrix
def plot_matrix(xaxis, yaxis, w, pdf, xlabel, ylabel, clabel, text=None):
    # the matrix values
    x = [xa for xa in xaxis for ya in yaxis]
    y = [ya for xa in xaxis for ya in yaxis]
    
    # the binning
    xbins = np.concatenate((xaxis - (xaxis[1] - xaxis[0]) / 2, [xaxis[-1] + (xaxis[1] - xaxis[0]) / 2]))
    ybins = np.concatenate((yaxis - (yaxis[1] - yaxis[0]) / 2, [yaxis[-1] + (yaxis[1] - yaxis[0]) / 2]))
    
    # plot the matrix
    fig, ax = plt.subplots(figsize=(6 * (len(xaxis) / len(yaxis)), 6))
    *_, image = ax.hist2d(x, y, weights=w, bins=(xbins, ybins), cmap='plasma_r')
    ax.set_xlabel(fr'{xlabel}')
    ax.set_ylabel(fr'{ylabel}')
    ax.get_yaxis().set_label_coords(-0.1, 0.5)
    
    # create the color axis
    cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.03, ax.get_position().height])
    cax.get_yaxis().set_label_coords(9.5 - len(xaxis) / 4, 0.5)
    
    # create the color bar
    cbar = plt.colorbar(image, cax=cax)
    cbar.set_label(clabel)
    
    # include a label if supplied
    if text is not None:
        ratio = fig.get_size_inches()[1] / fig.get_size_inches()[0]
        xpos  = ax.get_xlim()[1] - (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.04 * ratio
        ypos  = ax.get_ylim()[1] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.04
        ax.text(xpos, ypos, text, va='top', ha='right')
        
    # save the figure
    pdf.savefig(bbox_inches='tight')
    plt.close()

# calculate the resolution
def find_fwhm(data, window, n_slope=0):
    # the resolution information
    resol = 0
    error = 0
    extra = [np.nan, np.nan, np.nan, np.nan]
    
    # loop over the possible bin widths
    for i in range(20):
        # the binning for each point
        bins = np.arange(*window, 0.4 + 0.1 * i)

        # histogram the energy
        n = numba_histogram(data, bins)
        
        # find the largest bin
        i_max = np.argmax(n)
        
        # find the peak height
        peak, height = find_peak(data, bins)
    
        # find the first and last bin over threshold
        i_lo = i_max - np.argmax(n[:i_max+1][::-1] < height / 2) + 1
        i_hi = i_max + np.argmax(n[i_max:  ]       < height / 2) - 1
        
        # check that the fit range is reasonable
        if i_lo + n_slope + 1 >= i_max+1 or\
           i_hi - n_slope     <= i_max:
            # try the next bin width
            continue
        
        # find the lower range of bins to be fit
        i_lo_lo = i_lo - n_slope - 1
        i_lo_hi = i_lo + n_slope + 1
        
        # find the upper range of bins to be fit
        i_hi_lo = i_hi - n_slope
        i_hi_hi = i_hi + n_slope + 2
        
        # check that the fit range is reasonable
        if i_lo_lo <  0                                 or\
           i_hi_hi >= len(n)                            or\
           not np.all(np.diff(n[i_lo_lo:i_max+1]) >= 0) or\
           not np.all(np.diff(n[i_max  :i_hi_hi]) <= 0) or\
           n[i_lo_lo] < 30                              or\
           n[i_hi_hi] < 30:
            # try the next bin width
            continue
            
        # the bin centers
        c = (bins[:-1] + bins[1:]) / 2

        # the threshold
        v_th = height / 2
        d_th = height / 4
        
        try:
            # calculate the lower crossing
            wts = 1 / np.sqrt(n[i_lo_lo:i_lo_hi])
            (m, b), cov = np.polyfit(c[i_lo_lo:i_lo_hi], n[i_lo_lo:i_lo_hi], 1, w=wts, cov='unscaled')
            v_lo = (v_th - b) / m
            d_lo = (cov[0, 0] / np.power(m, 2) + (cov[1, 1] + d_th) / np.power(v_th - b, 2) +\
                2 * cov[0, 1] / (v_th - b) / m) * np.power(v_lo, 2)

            # calculate the upper crossing
            wts = 1 / np.sqrt(n[i_hi_lo:i_hi_hi])
            (m, b), cov = np.polyfit(c[i_hi_lo:i_hi_hi], n[i_hi_lo:i_hi_hi], 1, w=wts, cov='unscaled')
            v_hi = (v_th - b) / m
            d_hi = (cov[0, 0] / np.power(m, 2) + (cov[1, 1] + d_th) / np.power(v_th - b, 2) +\
                2 * cov[0, 1] / (v_th - b) / m) * np.power(v_hi, 2)
        except:
            # try the next bin width
            continue
        
        # the full-width half-max
        v_res = v_hi - v_lo
        d_res = np.sqrt(d_lo + d_hi)

        # check if the result is reasonable
        if np.isfinite(v_res) and\
           v_res >  1         and\
           v_res < 10:
            # update the result
            resol = v_res
            error = d_res
            extra = [v_lo, v_hi, peak, height]
            
            # skip the rest
            break
            
    # return the resolution information
    return bins, resol, error, extra
    