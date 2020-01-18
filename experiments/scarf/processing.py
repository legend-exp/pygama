#!/usr/bin/env python3
import os
import time
import h5py
import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt

from pygama import DataSet, read_lh5, get_lh5_header
import pygama.analysis.histograms as pgh

def main():
    """
    this is the high-level part of the code, something that a user might
    write (even on the interpreter) for processing with a specific config file.
    """

    #set the run index (first number in ds field in config.json) here:
    run_index = 3
    process_data(run_index)
    
    
    
    #plot_data()
    #plot_waveforms()
    

def process_data(run_index):
    from pygama import DataSet
    ds = DataSet(run_index, md="config.json")
    ds.daq_to_raw(overwrite=True, test=False)
    # ds.raw_to_dsp(....)


def plot_data():
    """
    read the lh5 output.
    """
    
    f_lh5 = "/Users/wisecg/Data/L200/tier1/t1_run0.lh5"
    df = get_lh5_header(f_lh5)
    
    # df = read_lh5(f_lh5)
    # print(df)
    exit()
    
    
    
    # hf = h5py.File("/Users/wisecg/Data/L200/tier1/t1_run0.lh5")
    
    # # 1. energy histogram
    # wf_max = hf['/daqdata/wf_max'][...] # slice reads into memory
    # wf_bl = hf['/daqdata/baseline'][...]
    # wf_max = wf_max - wf_bl
    # xlo, xhi, xpb = 0, 5000, 10
    # hist, bins = pgh.get_hist(wf_max, range=(xlo, xhi), dx=xpb)
    # plt.semilogy(bins, hist, ls='steps', c='b')
    # plt.xlabel("Energy (uncal)", ha='right', x=1)
    # plt.ylabel("Counts", ha='right', y=1)
    # # plt.show()
    # # exit()
    # plt.cla()
    
    # 2. energy vs time
    # ts = hf['/daqdata/timestamp']
    # plt.plot(ts, wf_max, '.b')
    # plt.show()
    
    # 3. waveforms
    nevt = hf['/daqdata/waveform/values/cumulative_length'].size
    
    # create a waveform block compatible w/ pygama
    # and yeah, i know, for loops are inefficient. i'll optimize when it matters
    wfs = []
    wfidx = hf["/daqdata/waveform/values/cumulative_length"] # where each wf starts
    wfdata = hf["/daqdata/waveform/values/flattened_data"] # adc values
    wfsel = np.arange(2000)
    for iwf in wfsel: 
        ilo = wfidx[iwf]
        ihi = wfidx[iwf+1] if iwf+1 < nevt else nevt
        wfs.append(wfdata[ilo : ihi])
    wfs = np.vstack(wfs)
    print(wfs.shape) # wfs on each row.  will work w/ pygama.

    # plot waveforms, flip polarity for fun
    for i in range(wfs.shape[0]):
        wf = wfs[i,:]
        plt.plot(np.arange(len(wf)), wf)
        
    plt.xlabel("clock ticks", ha='right', x=1)
    plt.ylabel("adc", ha='right', y=1)
    plt.tight_layout()
    plt.show()
    # plt.savefig(f"testdata_evt{ievt}.png")
    
    hf.close()


    
    
if __name__=="__main__":
    main()
