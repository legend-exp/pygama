#!/usr/bin/env python3
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('../../pygama/clint.mpl')

import pygama.io.lh5 as lh5
import pygama.analysis.histograms as pgh

def main():
    """
    tasks:
    - create combined dsp file (w/ onbd E and timestamp)
    - calibrate one run (onboard E, pygama trapE)
    - show resolution of 1460, 238, etc peaks
    - low-energy noise analysis (use dsp params)
    - determine pz correction value for OPPI
    """
    show_groups()
    # show_spectra()
    # simple_calibration()
    # show_wfs()
    
    
def show_groups():
    """
    show example of accessing the names of the HDF5 groups in our LH5 files
    """
    f_raw = '/Users/wisecg/Data/OPPI/raw/oppi_run0_cyc2027_raw.lh5'    
    # f_dsp = '/Users/wisecg/Data/OPPI/dsp/oppi_run0_cyc2027_dsp.lh5'    
    f_dsp = '/Users/wisecg/Data/OPPI/dsp/oppi_run0_cyc2027_dsp_test.lh5'
    
    # h5py method
    # hf = h5py.File(f_raw)
    # hf = h5py.File(f_dsp)
    
    # some examples of navigating the groups
    # print(hf.keys())
    # print(hf['ORSIS3302DecoderForEnergy/raw'].keys())
    # print(hf['ORSIS3302DecoderForEnergy/raw/waveform'].keys())
    # exit()
    
    # lh5 method
    sto = lh5.Store()
    groups = sto.ls(f_dsp)
    data = sto.read_object('ORSIS3302DecoderForEnergy/raw', f_dsp)
    
    for col in data.keys():
        print(col, data[col].nda.shape)
    
    # df_dsp = data.get_dataframe()
    # print(df_dsp.columns)


def show_spectra():
    """
    """
    f_dsp = '/Users/wisecg/Data/OPPI/dsp/oppi_run0_cyc2027_dsp.lh5'    
    
    # we will probably make this part simpler in the near future
    sto = lh5.Store()
    groups = sto.ls(f_dsp)
    data = sto.read_object('raw', f_dsp)
    df_dsp = data.get_dataframe()

    # from here, we can use standard pandas to work with data
    print(df_dsp)
    
    # one example: create uncalibrated energy spectrum,
    # using a pygama helper function to get the histogram
    
    elo, ehi, epb = 0, 100000, 10
    ene_uncal = df_dsp['trapE']
    hist, bins, _ = pgh.get_hist(ene_uncal, range=(elo, ehi), dx=epb)
    bins = bins[1:] # trim zero bin, not needed with ds='steps'

    plt.semilogy(bins, hist, ds='steps', c='b', label='trapE')
    plt.xlabel('trapE', ha='right', x=1)
    plt.ylabel('Counts', ha='right', y=1)
    plt.legend()
    plt.tight_layout()
    plt.show()
    

def simple_calibration():
    """
    get calibration constants for onbd energy and 'trapE' energy
    """
    f_dsp = '/Users/wisecg/Data/OPPI/dsp/oppi_run0_cyc2027_dsp.lh5'    
    
    
def show_wfs():
    """
    show low-e waveforms.  might make a separate file with a cut
    """
    f_raw = '/Users/wisecg/Data/OPPI/raw/oppi_run0_cyc2027_raw.lh5'
    
    
def run_dsp():
    """
    run dsp to a temporary file, try out new ProcessingChain parameters for low-e
    """
    f_raw = '/Users/wisecg/Data/OPPI/raw/oppi_run0_cyc2027_raw.lh5'
    
    
    
if __name__=="__main__":
    main()