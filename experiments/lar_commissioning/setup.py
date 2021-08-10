# import os module
import os

# turn off file locking
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# import python modules
import collections
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

# silence warnings if requested

warnings.filterwarnings('ignore')

# setup the logging as requested
importlib.reload(logging)
prefix = ''
logging.basicConfig(format=f'{prefix}%(message)s', level=logging.INFO, datefmt='%H:%M:%S')
    
# set the path to pygama
pyg_dir = "../pygama/pygama"
# update the system path
if pyg_dir  != '':
    sys.path = [os.getcwd() + '/' + pyg_dir ] + sys.path

# import pygama modules
import pygama.analysis.histograms as ph
import pygama.analysis.peak_fitting as pf
import pygama.dsp.build_processing_chain as bpc
import pygama.dsp.dsp_optimize as dspo
import pygama.dsp.processors as proc
import pygama.dsp.ProcessingChain as ProcessingChain
import pygama.dsp.units as units
import pygama.lh5 as lh5


