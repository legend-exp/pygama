import pandas as pd
import sys
import numpy as np
import scipy as sp
import json
import os
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

    deltaT()

def deltaT():

    if(len(sys.argv) != 2):
        print('Usage: dead_time.py [run number]')
        sys.exit()

    df = pd.read_hdf('~/Data/MJ60/pygama/t2_run'+sys.argv[1]+'.h5', columns=['timestamp'])
    
    df = df.reset_index(drop=True)
    df = df.loc[(df.index<32000)]

    df['permutated_timestamp'] = [0]*len(df)

    for i in range(0, len(df)-1):
        a = int(i)+1
        df['permutated_timestamp'][a] = df['timestamp'][i]

    df['deltaT'] = df['timestamp'] - df['permutated_timestamp']

    plt.hist((df['deltaT']/100e06)*1e06, np.arange(0,(2000000/100e06)*1e06,(1000/100e06)*1e06), histtype='step', color = 'black', label='30 microsecond minimum')
    plt.xlabel('Time Between Events (microseconds)', ha='right', x=1.0)
    plt.ylabel('Counts', ha='right', y=1.0)
    plt.tight_layout()
    plt.legend(frameon=True, loc='upper right', fontsize='small')
    plt.show()




if __name__ == '__main__':
        main()
