#!/usr/bin/env python3
from pygama import DataSet
from pprint import pprint

def main():
    """
    testing Yoann's Tier 0 (daq_to_raw) FlashCam parser.
    this is the high-level part of the code, something that a user might
    write (even on the interpreter) for processing with a specific config file.
    """
    process_data()
    # read_data()
    # test_cygama()
    

def process_data():
    from pygama import DataSet
    ds = DataSet(0, md="config.json")
    ds.daq_to_raw(overwrite=True, test=False)
    # ds.raw_to_dsp(....)


def read_data():
    """
    read the output
    """
    import h5py
    out_file = "/Users/wisecg/Data/L200/t1_run0.lh5"
    hf = h5py.File(out_file)
    
    header = hf['/header']
    for name, val in header.attrs.items():
        print(name, val)
    
    print(hf.keys())
    
    # the h5py book tells you not to try and loop over keys,
    # you should use "visit" or "visititems"
    def printname(name):
        print(name)
        
    def printobj(name, obj):
        print(name, obj)
    
    # hf.visit(printname)
    # hf.visititems(printobj)
    
    # get the number of entries in the dataset
    def getentries(name, obj):
        print(name, obj)
        # if isinstance(obj, h5py.Dataset):
            # print("size is:", obj.shape)
    hf.visititems(getentries)
    # exit()
    
    # --------------------------------------------------------------------------
    # 1. energy histogram
    import pygama.analysis.histograms as pgh
    import matplotlib.pyplot as plt
    plt.style.use("../../pygama/clint.mpl")
    
    
    wf_max = hf['/daqdata/wf_max'][...] # slice reads into memory
    wf_bl = hf['/daqdata/baseline'][...]
    wf_max = wf_max - wf_bl
    xlo, xhi, xpb = 0, 5000, 10
    hist, bins = pgh.get_hist(wf_max, range=(xlo, xhi), dx=xpb)
    plt.semilogy(bins, hist, ls='steps', c='b')
    plt.xlabel("Energy (uncal)", ha='right', x=1)
    plt.ylabel("Counts", ha='right', y=1)
    # plt.show()
    # exit()
    plt.cla()
    
    # 2. energy vs time
    # ts = hf['/daqdata/timestamp']
    # plt.plot(ts, wf_max, '.b')
    # plt.show()
    
    # 3. waveforms
    import numpy as np
    nevt = hf['/daqdata/ievt'].size
    
    print('nevt:', nevt)
    
    # create a waveform block compatible w/ pygama
    # and yeah, i know, for loops are inefficient. i'll optimize when it matters
    wfs = []
    wfidx = hf["/daqdata/waveform/cumulative_length"] # where each wf starts
    wfdata = hf["/daqdata/waveform/flattened_data"] # adc values
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


def test_cygama():
    """
    """
    import pygama.cygama as pc
    print(pc.add(1,2))
    
    
if __name__=="__main__":
    main()