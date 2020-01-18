#!/usr/bin/env python3
import os
import time
import h5py
import sys
import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt

from pygama import DataSet, read_lh5, get_lh5_header
import pygama.analysis.histograms as pgh

# new viewer for waveforms for llama / scarf tests.
# Largely "inspired" by Clint's processing.py

def main():
    """
    Currently this is quite simple: just take a filename from argument
    (ignoring the whole fancy run number stuff) and throws it to the plotter
    """
    #process_data()
    
    #plotWFS()
    #exit(0)

    
    try:
        filename = sys.argv[1]
    except:
        print("You have to give a file name as argument!")
        exit(0)
    
    plotWFS(filename)
    #plot_data(filename)
    #testmethod(filename)




def plotWFS(filename):   
    print("enter channels to plot, e.g. 0 2 3 for plotting channels 0, 2 and 3 at the same time.")
    user = input("enter channels> ")
    channels = list(map(int,user.strip().split()))
    print("will plot the following channels:")
    print(channels)
    hf = h5py.File(filename, "r")
    nevt = hf['/daqdata/waveform/values/cumulative_length'].size
    wfs = []
    wfidx = hf["/daqdata/waveform/values/cumulative_length"] # where each wf starts
    wfdata = hf["/daqdata/waveform/values/flattened_data"] # adc values
    chunksize = 2000 if nevt > 2000 else nevt - 1
    wfsel = np.arange(chunksize)
    for iwf in wfsel: 
        ilo = wfidx[iwf]
        ihi = wfidx[iwf+1] if iwf+1 < nevt else nevt
        wfs.append(wfdata[ilo : ihi])
    wfs = np.vstack(wfs)

    plt.ion()   #make plot non-blocking

    while True:
        index = int(input("enter index> "))
        wfx = getWFEvent(hf, wfs, index, channels)
        plt.clf()
        for q in wfx:
            plt.plot(q)
        #for i in range(wfx.shape[0]):
        #    wf = wfx[i,:]
        #    plt.plot(np.arange(len(wf)), wf)

        plt.tight_layout()
        plt.show()
        plt.pause(0.001)

    hf.close()

def getWFEvent(hf, waveforms, index, channels):
    current = [0] * len(channels)
    i = 0
    chIDs = hf["/daqdata/channel"]
    wfList = []
    while True:
        try:
            #print(i, chIDs[i], channels.index(chIDs[i]))
            indexx = channels.index(chIDs[i])
        except ValueError as e:
            i+=1
            continue
        if current[indexx] == index:
            wfList.append(waveforms[i])
        current[indexx] += 1
        i += 1
        if len(wfList) == len(channels):
            return wfList




def testmethod(filename):
    hf = h5py.File(filename)
    ts = hf['/daqdata/timestamp']
    print(ts.size, ts[137], ts[138], ts[707])
    #here data from the daw is stored
    print(hf['/daqdata'])
    #here header info is stored --> stuff that does not change btw events
    # see http://docs.h5py.org/en/stable/high/attr.html#attributes
    print(hf['/header'].attrs.get("file_name"), hf['/header'].attrs.get("nsamples"))
    print(hf['/header'].attrs.get("file_name"))
    print("The following metadata is available in the header:")
    print(hf['/header'].attrs.keys())
    plt.plot(ts)
    plt.show()

def test2():
    while(True):
        user = input("tell me something> ")
        print(user)
        li = list(map(int,user.strip().split()))
        print(li[2])


def plot_data(filename):
    """
    read the lh5 output.
    plot waveforms
    Mostly written by Clint
    """
    
    

    #filename = "/mnt/e15/schwarz/testdata_pg/scarf/tier1/t1_run2002.lh5"
    df = get_lh5_header(filename)
    
    #df = read_lh5(filename)
    print(df)
    #exit()
    
    
    
    hf = h5py.File(filename)
    
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
    print("Shape of the waveforms:")
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
