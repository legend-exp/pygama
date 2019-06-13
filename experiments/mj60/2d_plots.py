import pandas as pd
import sys
import time
import numpy as np
import scipy as sp
import scipy.optimize as opt
import scipy.signal as signal
import os, json
import pygama.dataset as ds
import pygama.analysis.histograms as pgh
import pygama.dsp.transforms as pgt
import pygama.utils as pgu
import pygama.analysis.peak_fitting as pga
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
plt.style.use('style.mplstyle')

def main():

    #ttrap_max_vs_energy()
    #rise()
    rise_diff()

def ttrap_max_vs_energy():

    if(len(sys.argv) != 2):
        print('Usage: 2d_plots.py [run number]')
        sys.exit()

    start = time.time()

    with open("runDB.json") as f:
        runDB = json.load(f)
    tier_dir = os.path.expandvars(runDB["tier_dir"])
    meta_dir = os.path.expandvars(runDB["meta_dir"])

    df = pd.read_hdf('{}/t2_run{}.h5'.format(tier_dir,sys.argv[1]))
    df['e_cal'] = pd.read_hdf('{}/Spectrum_{}_2.hdf5'.format(meta_dir,sys.argv[1]))['e_cal']

    plt.hist2d(df['e_cal'], df['ttrap_max']/df['e_cal'], np.arange(0,100,0.1), norm=LogNorm())
    plt.xlim(0,100)
    plt.ylim(0,5)
    plt.xlabel('Energy (keV)', ha='right', x=1.0)
    plt.ylabel('ttrap_max/E', ha='right', y=1.0)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts')
    plt.tight_layout()
    plt.show()

def rise():

    if(len(sys.argv) != 2):
        print('Usage: 2d_plots.py [run number]')
        sys.exit()

    with open("runDB.json") as f:
        runDB = json.load(f)
    tier_dir = os.path.expandvars(runDB["tier_dir"])
    meta_dir = os.path.expandvars(runDB["meta_dir"])

    df = pd.read_hdf('{}/t2_run{}.h5'.format(tier_dir,sys.argv[1]))
    df['e_cal'] = pd.read_hdf('{}/Spectrum_{}_2.hdf5'.format(meta_dir,sys.argv[1]))['e_cal']
    df['rise'] = df['tp100'] - df['t0']

    plt.hist2d(df['e_cal'], df['rise'], np.arange(0,1500,0.5), norm=LogNorm())
    plt.xlim(0,50)
    plt.ylim(0,1500)
    plt.xlabel('Energy (keV)', ha='right', x=1.0)
    plt.ylabel('tp100 - t0', ha='right', y=1.0)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts')
    plt.tight_layout()
    plt.show()

def rise_diff():

    if(len(sys.argv) != 3):
        print('Usage: 2d_plots.py [run number 1] [run number 2]')
        sys.exit()

    with open("runDB.json") as f:
        runDB = json.load(f)
    tier_dir = os.path.expandvars(runDB["tier_dir"])
    meta_dir = os.path.expandvars(runDB["meta_dir"])

    df = pd.read_hdf('{}/t2_run{}.h5'.format(tier_dir,sys.argv[1]))
    df['e_cal'] = pd.read_hdf('{}/Spectrum_{}_2.hdf5'.format(meta_dir,sys.argv[1]))['e_cal']
    df['rise'] = df['tp100'] - df['t0']

    df_2 = pd.read_hdf('{}/t2_run{}.h5'.format(tier_dir,sys.argv[2]))
    df_2['e_cal'] = pd.read_hdf('{}/Spectrum_{}_2.hdf5'.format(meta_dir,sys.argv[2]))['e_cal']
    df_2['rise'] = df_2['tp100'] - df_2['t0']

    xlo, xhi, xpb = 0, 50, 2
    ylo, yhi, ypb = 0, 1400, 10
     
    nxbins = int((xhi-xlo)/xpb)
    nybins = int((yhi-ylo)/ypb)

    hist1, xbins, ybins = np.histogram2d(df['e_cal'], df['rise'], [nxbins,nybins], [[xlo,xhi], [ylo,yhi]])
    hist2, xbins, ybins = np.histogram2d(df_2['e_cal'], df_2['rise'], [nxbins,nybins], [[xlo,xhi], [ylo,yhi]])

    hist_diff = hist2 - hist1

    xbins = xbins[0:(len(xbins)-1)]
    ybins = ybins[0:(len(ybins)-1)]

    plt.hist2d(hist_diff[0],hist_diff[1], [xbins, ybins])
    plt.xlim(xlo, xhi)
    plt.ylim(ylo, yhi)
    plt.xlabel('Energy (keV)', ha='right', x=1.0)
    plt.ylabel('tp100 - t0', ha='right', y=1.0)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
        main()
