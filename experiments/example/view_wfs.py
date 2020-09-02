#!/usr/bin/env python3
import os
import time
import h5py
import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt

import pygama.analysis.histograms as pgh

def main():
    """
    """
    hf = h5py.File("/Users/wisecg/Data/lh5/t1_run1.lh5")
    
    # some examples of navigating the groups
    # print(hf.keys())
    # print(hf['daqdata'].keys())
    # print(hf['daqdata/waveform'].keys())
    # exit()
    
    # # 1. energy histogram
    # wf_max = hf['/daqdata/wf_max'][...] # the '...' reads into memory
    # wf_bl = hf['/daqdata/baseline'][...]
    # wf_max = wf_max - wf_bl
    # xlo, xhi, xpb = 0, 5000, 10
    # hist, bins, var = pgh.get_hist(wf_max, range=(xlo, xhi), dx=xpb)
    # plt.semilogy(bins[1:], hist, ds='steps', c='b')
    # plt.xlabel("Energy (uncal)", ha='right', x=1)
    # plt.ylabel("Counts", ha='right', y=1)
    # plt.tight_layout()
    # plt.show()
    # exit()
    # plt.cla()
    
    # # 2. energy vs time
    # wf_max = hf['/daqdata/wf_max'][...]
    # ts = hf['/daqdata/timestamp']
    # plt.plot(ts, wf_max, '.b')
    # plt.show()
    # plt.cla()
    # exit()
    
    # 3. waveforms
    nevt = hf['/daqdata/waveform/values/cumulative_length'].size
    
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