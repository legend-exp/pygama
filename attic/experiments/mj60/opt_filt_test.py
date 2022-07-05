import sys
import os, json
import numpy as np
import scipy as sp
import pandas as pd
import scipy.optimize as opt
from scipy.signal import medfilt, find_peaks
from pygama import DataSet
import pygama.analysis.histograms as pgh
import pygama.utils as pgu
import pygama.analysis.peak_fitting as pga
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.style.use('style.mplstyle')
np.set_printoptions(threshold=np.inf)

def main():

    cut_t1()


def cut_t1():

    with open("runDB.json") as f:
        runDB = json.load(f)
    tier_dir = os.path.expandvars(runDB["tier_dir"])

    df = pd.read_hdf('{}/t1_run280.h5'.format(tier_dir), '/ORSIS3302DecoderForEnergy')
    df_2 =  pd.read_hdf("{}/t2_run280.h5".format(tier_dir))

    df['e_cal'] = 0.4054761904761905 * df_2['e_ftp'] + 3.113095238095184
    
    df = df.loc[(df.e_cal>1410)&(df.e_cal<1510)]
    
    df.to_hdf('{}/t1_run0.h5'.format(tier_dir), key='df', mode='w')

if __name__ == '__main__':
        main()
