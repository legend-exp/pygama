import numpy as np
import h5py
import pandas as pd
import sys
import json
import os
from pygama import DataSet
import pygama.dataset as ds
import matplotlib.pyplot as plt
plt.style.use('style.mplstyle')

def main():

    #Campaign_Data()
    #Kr_and_BKG_Data()
    Collimator_Simulations()

def Campaign_Data():

    with open("runDB.json") as f:
        runDB = json.load(f)
    meta_dir = os.path.expandvars(runDB["meta_dir"])

    BKG1 =  pd.read_hdf("{}/Spectrum_280-289.hdf5".format(meta_dir))
    Kr1 = pd.read_hdf("{}/Spectrum_330-339.hdf5".format(meta_dir))

    xlo, xhi, xpb = 0, 4000, 0.5
    nbins = int((xhi - xlo)/xpb)

    BKGhist, bins = np.histogram(BKG1['e_cal'], nbins, (xlo,xhi))
    Krhist, bins = np.histogram(Kr1['e_cal'], nbins, (xlo,xhi))

    bins = bins[0:(len(bins)-1)]
    bin_centers = bins - (bins[1] - bins[0])/2

    integral1 = xpb * sum(BKGhist[40:2600])
    integral2 = xpb * sum(Krhist[40:2600])

    hist_01 = BKGhist * integral2/integral1
    hist3 = Krhist - hist_01
    errors = np.sqrt(Krhist + hist_01*integral2/integral1)

    plt.plot(bins, hist_01, color='red', ls='steps', label='Background Data')
    plt.plot(bins, Krhist, color='black', ls='steps', label='Kr83m Data')
    plt.plot(bins, hist3, color='aqua', ls='steps', label='Kr83m Spectrum (Kr83m Data - Background Data)')
    plt.errorbar(bin_centers, hist3, errors, color='black', fmt='o', markersize=3, capsize=3)

    plt.xlim(0,4000)
    #plt.ylim(0,plt.ylim()[1])
    plt.xlabel('Energy (keV)', ha='right', x=1.0)
    plt.ylabel('Counts', ha='right', y=1.0)
    plt.title('Energy Spectrum (Kr83m Source)')
    plt.legend(frameon=True, loc='upper right', fontsize='small')
    plt.tight_layout()
    #plt.semilogy()
    plt.show()

def Kr_and_BKG_Data():

    with open("runDB.json") as f:
        runDB = json.load(f)
    meta_dir = os.path.expandvars(runDB["meta_dir"])
    
    BKG1 =  pd.read_hdf("{}/Spectrum_101.hdf5".format(meta_dir), key="df")
    BKG2 =  pd.read_hdf("{}/Spectrum_105.hdf5".format(meta_dir), key="df")
    BKG3 =  pd.read_hdf("{}/Spectrum_106.hdf5".format(meta_dir), key="df")
    BKG4 =  pd.read_hdf("{}/Spectrum_107.hdf5".format(meta_dir), key="df")

    Kr1 =  pd.read_hdf("{}/Spectrum_103.hdf5".format(meta_dir), key="df")
    Kr2 =  pd.read_hdf("{}/Spectrum_104.hdf5".format(meta_dir), key="df")
    Kr3 =  pd.read_hdf("{}/Spectrum_109.hdf5".format(meta_dir), key="df")
    Kr4 =  pd.read_hdf("{}/Spectrum_110.hdf5".format(meta_dir), key="df")
    Kr5 =  pd.read_hdf("{}/Spectrum_111.hdf5".format(meta_dir), key="df")
    Kr6 =  pd.read_hdf("{}/Spectrum_112.hdf5".format(meta_dir), key="df")
    Kr7 =  pd.read_hdf("{}/Spectrum_143.hdf5".format(meta_dir), key="df")
    Kr8 =  pd.read_hdf("{}/Spectrum_144.hdf5".format(meta_dir), key="df")
    Kr9 =  pd.read_hdf("{}/Spectrum_145.hdf5".format(meta_dir), key="df")
    Kr10 =  pd.read_hdf("{}/Spectrum_146.hdf5".format(meta_dir), key="df")
    Kr11 =  pd.read_hdf("{}/Spectrum_147.hdf5".format(meta_dir), key="df")
    Kr12 =  pd.read_hdf("{}/Spectrum_148.hdf5".format(meta_dir), key="df")
    Kr13 =  pd.read_hdf("{}/Spectrum_149.hdf5".format(meta_dir), key="df")
    Kr14 =  pd.read_hdf("{}/Spectrum_150.hdf5".format(meta_dir), key="df")
    Kr15 =  pd.read_hdf("{}/Spectrum_152.hdf5".format(meta_dir), key="df")
    Kr16 =  pd.read_hdf("{}/Spectrum_153.hdf5".format(meta_dir), key="df")
    Kr17 =  pd.read_hdf("{}/Spectrum_154.hdf5".format(meta_dir), key="df")
    Kr18 =  pd.read_hdf("{}/Spectrum_155.hdf5".format(meta_dir), key="df")
    Kr19 =  pd.read_hdf("{}/Spectrum_160.hdf5".format(meta_dir), key="df")
    Kr20 =  pd.read_hdf("{}/Spectrum_168.hdf5".format(meta_dir), key="df")

    xlo, xhi, xpb = 0, 4000, 0.5
    nbins = int((xhi - xlo)/xpb)

    BKGhist1, bins = np.histogram(BKG1['e_cal'], nbins, (xlo,xhi))
    BKGhist2, bins = np.histogram(BKG2['e_cal'], nbins, (xlo,xhi))
    BKGhist3, bins = np.histogram(BKG3['e_cal'], nbins, (xlo,xhi))
    BKGhist4, bins = np.histogram(BKG4['e_cal'], nbins, (xlo,xhi))

    BKGhist = BKGhist1 + BKGhist2 + BKGhist3 + BKGhist4

    Krhist1, bins = np.histogram(Kr1['e_cal'], nbins, (xlo,xhi))
    Krhist2, bins = np.histogram(Kr2['e_cal'], nbins, (xlo,xhi))
    Krhist3, bins = np.histogram(Kr3['e_cal'], nbins, (xlo,xhi))
    Krhist4, bins = np.histogram(Kr4['e_cal'], nbins, (xlo,xhi))
    Krhist5, bins = np.histogram(Kr5['e_cal'], nbins, (xlo,xhi))
    Krhist6, bins = np.histogram(Kr6['e_cal'], nbins, (xlo,xhi))
    Krhist7, bins = np.histogram(Kr7['e_cal'], nbins, (xlo,xhi))
    Krhist8, bins = np.histogram(Kr8['e_cal'], nbins, (xlo,xhi))
    Krhist9, bins = np.histogram(Kr9['e_cal'], nbins, (xlo,xhi))
    Krhist10, bins = np.histogram(Kr10['e_cal'], nbins, (xlo,xhi))    
    Krhist11, bins = np.histogram(Kr11['e_cal'], nbins, (xlo,xhi))
    Krhist12, bins = np.histogram(Kr12['e_cal'], nbins, (xlo,xhi))
    Krhist13, bins = np.histogram(Kr13['e_cal'], nbins, (xlo,xhi))
    Krhist14, bins = np.histogram(Kr14['e_cal'], nbins, (xlo,xhi))
    Krhist15, bins = np.histogram(Kr15['e_cal'], nbins, (xlo,xhi))
    Krhist16, bins = np.histogram(Kr16['e_cal'], nbins, (xlo,xhi))
    Krhist17, bins = np.histogram(Kr17['e_cal'], nbins, (xlo,xhi))
    Krhist18, bins = np.histogram(Kr18['e_cal'], nbins, (xlo,xhi))
    Krhist19, bins = np.histogram(Kr19['e_cal'], nbins, (xlo,xhi))
    Krhist20, bins = np.histogram(Kr20['e_cal'], nbins, (xlo,xhi))

    #Krhist = Krhist1 + Krhist2 + Krhist3 + Krhist4 + Krhist5 + Krhist6 + Krhist7 + Krhist8 + Krhist9 + Krhist10 + Krhist11 + Krhist12 + Krhist13 + Krhist14 + Krhist15 +Krhist16 + Krhist17 + Krhist18 + Krhist19 + Krhist20

    Krhist = Krhist8 + Krhist9 + Krhist10 + Krhist15 + Krhist20

    bins = bins[0:(len(bins)-1)]
    bin_centers = bins - (bins[1] - bins[0])/2

    integral1 = xpb * sum(BKGhist[40:2600])
    integral2 = xpb * sum(Krhist[40:2600])

    hist_01 = BKGhist * integral2/integral1
    hist3 = Krhist - hist_01
    errors = np.sqrt(Krhist + hist_01*integral2/integral1)

    plt.plot(bins, hist_01, color='red', ls='steps', label='Background Data')
    plt.plot(bins, Krhist, color='black', ls='steps', label='Kr83m Data')
    plt.plot(bins, hist3, color='aqua', ls='steps', label='Kr83m Spectrum (Kr83m Data - Background Data)')
    plt.errorbar(bin_centers, hist3, errors, color='black', fmt='o', markersize=3, capsize=3)

    plt.xlim(0,4000)
    #plt.ylim(0,plt.ylim()[1])
    plt.xlabel('Energy (keV)', ha='right', x=1.0)
    plt.ylabel('Counts', ha='right', y=1.0)
    plt.title('Energy Spectrum (Kr83m Source)')
    plt.legend(frameon=True, loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.semilogy()
    plt.show()

def Collimator_Simulations():

    if(len(sys.argv) != 3):
        print('Usage: spectra.py [run number] [source]')
        sys.exit()

    with open("runDB.json") as f:
        runDB = json.load(f)
    meta_dir = os.path.expandvars(runDB["meta_dir"])

    # A_0 = activity of source in mCi
    A_0 = .005

    # A = activity of source in bequerel (decays/second)
    A = A_0*37000000  
    print("activity of source in bequerel = {}".format(A))
 
    # B = number of primaries ran in g4simple
    B = 10000000

    df1 =  pd.read_hdf("{}/Spectrum_{}.hdf5".format(meta_dir,sys.argv[1]), key="df")
    df2 =  pd.read_hdf("{}/processed.hdf5".format(meta_dir), key="procdf")

    runtime = ds.DataSet(run=int(sys.argv[1]), md='./runDB.json').get_runtime()
    print('total runtime = {} seconds'.format(runtime))

    m = list(df2['energy'])
    p = list(x*1000 for x in m)

    xlo, xhi, xpb = 0, 4500, 5
    nbins = int((xhi - xlo)/xpb)

    hist1, bins = np.histogram(df1['e_cal'], nbins, (xlo,xhi))
    hist2, bins = np.histogram(p, nbins, (xlo,xhi))
    bins = bins[1:(len(bins))]
    
    hist_02 = (hist2/B)*A*runtime
    print("total number of simulation counts = {}".format(sum(hist2)))
    hist3 = hist1 + hist_02

    bkg_counts_per_second = (sum(hist1[0:4500])/runtime)

    counts_per_second = (sum(hist3[0:4500])/runtime)

    plt.plot(bins, hist_02, ls='steps', color='darkgreen', label=str(sys.argv[2])+' Simulation Data for a '+str(A_0)+' mCi Source')
    plt.plot(bins, hist3, ls='steps', color='black', label='Background Data + '+str(sys.argv[2])+' Simulation Data, \n {:.0f} counts per second'.format(counts_per_second))
    plt.plot(bins, hist1, ls='steps', color='purple', label='MJ60 Background Data, {:.0f} counts per second'.format(bkg_counts_per_second))
    plt.xlim(0,4000)
    #plt.ylim(0,plt.ylim()[1])
    plt.xlabel('Energy (keV)', ha='right', x=1.0)
    plt.ylabel('Counts', ha='right', y=1.0)
    plt.legend(frameon=True, loc='upper right', fontsize='small')
    plt.tight_layout()
    #plt.semilogx()
    plt.semilogy()
    plt.show()

if __name__ == '__main__':
        main()
