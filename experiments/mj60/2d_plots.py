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
    #rise_diff()
    AoverE_vs_E()
    #test()

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

    plt.hist2d(df['e_cal'], df['ttrap_max'], np.arange(0,100,1), norm=LogNorm())
    plt.xlim(0,50)
    plt.ylim(0,100)
    plt.xlabel('Energy (keV)', ha='right', x=1.0)
    plt.ylabel('ttrap_max', ha='right', y=1.0)
    plt.title('Kr83m Data')
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
    df['rise'] = df['tp50'] - df['t0']

    plt.hist2d(df['e_cal'], df['rise'], np.arange(-1000,1500,0.5), norm=LogNorm())
    plt.xlim(0,50)
    plt.ylim(-1000,1500)
    plt.xlabel('Energy (keV)', ha='right', x=1.0)
    plt.ylabel('tp50 - t0', ha='right', y=1.0)
    plt.title('Kr83m Data')
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

    #hist_diff = hist2 - hist1
    hist2 = hist2.T

    xbins = xbins[0:(len(xbins)-1)]
    ybins = ybins[0:(len(ybins)-1)]

    plt.hist2d(hist2[0],hist2[1], [xbins, ybins])
    plt.xlim(xlo, xhi)
    plt.ylim(ylo, yhi)
    plt.xlabel('Energy (keV)', ha='right', x=1.0)
    plt.ylabel('tp100 - t0', ha='right', y=1.0)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts')
    plt.tight_layout()
    plt.show()


def AoverE_vs_E():

    if(len(sys.argv) != 3):
        print('Usage: 2d_plots.py [run number 1] [run number 2]')
        sys.exit()

    start = time.time()

    with open("runDB.json") as f:
        runDB = json.load(f)
    tier_dir = os.path.expandvars(runDB["tier_dir"])
    meta_dir = os.path.expandvars(runDB["meta_dir"])

    # make 2D plot
    def plot_2D_hist():
        df = pd.read_hdf('{}/t2_run{}.h5'.format(tier_dir,sys.argv[1]))
        df['e_cal'] = pd.read_hdf('{}/Spectrum_{}.hdf5'.format(meta_dir,sys.argv[1]))['e_cal']
        plt.hist2d(df['e_cal'], (df['current_max']/df['e_cal']), bins=[100,200], range=[[0, 50], [0, .1]], normed=True, cmap='jet')
        plt.xlim(5,50)
        plt.xlabel('E (keV)', ha='right', x=1.0)
        plt.ylabel('A/E', ha='right', y=1.0)
        plt.title("Run {}".format(sys.argv[1]))
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Counts (normalized)')
        plt.tight_layout()
        plt.show() 

    # make 1D hist
    def plot_1D_hist(a,b):
        df = pd.read_hdf('{}/t2_run{}.h5'.format(tier_dir,sys.argv[1]))
        df['e_cal'] = pd.read_hdf('{}/Spectrum_{}.hdf5'.format(meta_dir,sys.argv[1]))['e_cal']
        df = df.loc[(df.e_cal>=float(a))&(df.e_cal<=float(b))]
        df_2 = pd.read_hdf('{}/t2_run{}.h5'.format(tier_dir,sys.argv[2]))
        df_2['e_cal'] = pd.read_hdf('{}/Spectrum_{}.hdf5'.format(meta_dir,sys.argv[2]))['e_cal']
        df_2 = df_2.loc[(df_2.e_cal>=float(a))&(df_2.e_cal<=float(b))]
        plt.hist(df['current_max']/df['e_cal'], np.arange(0,.2,.0010), histtype='step', density=True, label='run {}, {} < E < {} keV'.format(sys.argv[1],a,b))
        plt.hist(df_2['current_max']/df_2['e_cal'], np.arange(0,.2,.0010), histtype='step', density=True, label='run {}, {} < E < {} keV'.format(sys.argv[2],a,b))
        plt.xlabel('A/E', ha='right', x=1.0)
        plt.ylabel('Counts (normalized)', ha='right', y=1.0)
        plt.legend(frameon=True, loc='best', fontsize='small')
        plt.show()

    #plot_2D_hist()
    plot_1D_hist(a=25,b=30)


def test():

    if(len(sys.argv) != 3):
        print('Usage: 2d_plots.py [run number 1] [run number 2]')
        sys.exit()

    with open("runDB.json") as f:
        runDB = json.load(f)
    tier_dir = os.path.expandvars(runDB["tier_dir"])
    meta_dir = os.path.expandvars(runDB["meta_dir"])

    df = pd.read_hdf('{}/t2_run{}.h5'.format(tier_dir,sys.argv[1]))
    df['e_cal'] = pd.read_hdf('{}/Spectrum_{}_2.hdf5'.format(meta_dir,sys.argv[1]))['e_cal']
    df['rise'] = df['tp100'] - df['tp50']

    df_2 = pd.read_hdf('{}/t2_run{}.h5'.format(tier_dir,sys.argv[2]))
    df_2['e_cal'] = pd.read_hdf('{}/Spectrum_{}_2.hdf5'.format(meta_dir,sys.argv[2]))['e_cal']
    df_2['rise'] = df_2['tp100'] - df_2['tp50']

    x = df_2['e_cal']
    y = df_2['rise']

    x2 = df['e_cal']
    y2 = df['rise']

    f = plt.figure(figsize=(5,5))
    p1 = f.add_subplot(111, title='Kr83m - Background', xlabel='Energy (keV)', ylabel='tp100-tp50')
    h1,xedg1,yedg1 = np.histogram2d(x, y, bins=[25,25], range=[[0,50],[0,1400]])
    h2,xedg1,yedg1 = np.histogram2d(x2, y2, bins=[25,25], range=[[0,50],[0,1400]])
    h1 = h1.T
    h2 = h2.T
    h3 = h1 - h2
    #hMin, hMax = np.amin(h1), np.amax(h1)
    # # im1 = p1.imshow(h1,cmap='jet',vmin=hMin,vmax=hMax, aspect='auto') #norm=LogNorm())
    im1 = p1.imshow(h3,cmap='jet', origin='lower', aspect='auto', extent=[xedg1[0], xedg1[-1], yedg1[0], yedg1[-1]])

    cb1 = f.colorbar(im1, ax=p1, fraction=0.037, pad=0.04)

    #plt.hist2d(hist2[0],hist2[1], [xbins, ybins])
    #plt.xlim(xlo, xhi)
    #plt.ylim(ylo, yhi)
    #plt.xlabel('Energy (keV)', ha='right', x=1.0)
    #plt.ylabel('tp100 - t0', ha='right', y=1.0)
    #cbar = plt.colorbar()
    #cbar.ax.set_ylabel('Counts')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
        main()
