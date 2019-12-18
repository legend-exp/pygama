#!/usr/bin/env python3
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

def main():
    
    show_h5()
    # show_evts()
    # test_bool()
    write_h5()

    
def show_h5():
    """
    simple LEGEND data viewer function.
    shows the group structure, attributes, units, wfs, etc.
    can also do a lot of this from the cmd line with:
    $ h5ls -vlr [file]
    """
    hf = h5py.File("daq_testdata2.h5")

    for path, dset in h5iter(hf):
        print(path)

        # skip bool columns for now (see test_bool below)
        if any(x in path for x in ["inverted", "muveto"]):
            print("  skipping bool dataset:", path)
            continue
        
        dt = dset.dtype
        ds = dset.size
        at = dset.attrs
        ldtype = get_legend_dtype(dset.attrs['datatype']) 

        print(f"  type: {dt}  size: {ds}")
        for name in dset.attrs:
            print(f"  attr: {name}  value: {dset.attrs[name]}")
        for d in ldtype:
            print(f"    {d} : {ldtype[d]}")
        print("")


def h5iter(g, prefix=''):
    """
    simple iterator to expose groups and attributes of 
    everything in a given hdf5 file.  can also start at [prefix]-level.
    """
    for key in g.keys():
        item = g[key]
        path = '{}/{}'.format(prefix, key)
        if isinstance(item, h5py.Dataset): # test for dataset
            yield (path, item)
        elif isinstance(item, h5py.Group): # test for group (goes down a level)
            yield from h5iter (item, path)


def get_legend_dtype(dtstr):
    """
    the 'datatype' attribute is specific to LEGEND HDF5 files.
    reference: http://legend-exp.org/legend-data-format-specs/dev/
    right now, this fills an output dict by string splitting.
    there is probably a better way to read the Julia-formatted string.
    also, think about changing enums to numpy named tuples for faster lookups.
    """
    dt = {}
    
    # TODO: oliver gave me these, we should update the function below
    # to use them instead of all that messy string splitting
    # datatype_regexp = r"""^(([A-Za-z_]*)(<([0-9,]*)>)?)(\{(.*)\})?$"""
    # arraydims_regexp = r"""^<([0-9,]*)>$"""
    
    sp1 = dtstr.split("{")[0]
    dt["format"] = sp1 # scalar, array, struct, or table
    
    if "array" in dt["format"]:
        dt["ndim"] = int(sp1.split("<")[1].split(">")[0])

    sp2 = dtstr.split("{")[1:]
    if "enum" in sp2:
        dt["dtype"] = "enum"
        sp3 = sp2[1].split(",")
        for tmp in sp3:
            tmp = tmp.rstrip("}").split("=")
            dt[int(tmp[1])] = tmp[0]
    else:
        dt["dtype"] = sp2[0].rstrip("}")
    
    # pprint(dt)
    return dt


def show_evts():
    """
    look at events (make a small pandas dataframe).
    this is specific to Oliver's test file.  
    *** Let's worry about making this fast LATER. ***
    """
    hf = h5py.File("daq_testdata2.h5")
    data = hf['daqdata']
    nevt = data['evtno'].size
    cols = list(data.keys())

    # handle single-valued stuff
    tmp = []
    for c in cols:
            
        # skip bool columns for now (see test_bool below)
        if any(x in c for x in ["inverted", "muveto"]):
            continue
        
        # waveform data is handled below
        if not isinstance(data[c], h5py.Dataset):
            continue

        # for now, slice into the whole array and turn into pd.Series
        ser = pd.Series(data[c][...], name=c)
        tmp.append(ser)
        
    # combine the Series's into a DataFrame
    df = pd.concat(tmp, axis=1)
    print(df)
    
    # get waveform metadata
    wfdt = data["waveform_hf/dt"][0] # in the test file these are always the same
    wft0 = data["waveform_hf/t0"][0] 
    wfdt_unit = data["waveform_hf/dt"].attrs["units"]
    wft0_unit = data["waveform_hf/dt"].attrs["units"]
    print(f"wfdt: {wfdt} {wfdt_unit}  wft0: {wft0} {wft0_unit}")
    
    # show waveforms.  for fun, select waveforms from the first multi-hit event.
    ievt = 28
    evtno = df['daqevtno'].values
    wfsel = np.where(evtno == ievt)[0]

    # create a waveform block compatible w/ pygama
    # and yeah, i know, for loops are inefficient. i'll optimize when it matters
    wfs = []
    wfidx = data["waveform_hf/values/cumulative_length"] # where each wf starts
    wfdata = data["waveform_hf/values/flattened_data"] # adc values
    for iwf in wfsel: 
        ilo = wfidx[iwf]
        ihi = wfidx[iwf+1] if iwf+1 < nevt else nevt
        wfs.append(wfdata[ilo : ihi])
    wfs = np.vstack(wfs)
    print(wfs.shape) # wfs on each row.  will work w/ pygama.

    # plot waveforms, flip polarity for fun
    for i in range(wfs.shape[0]):
        wf = wfs[i,:]
        plt.plot(np.arange(len(wf)), wf * -1)
        
    plt.xlabel("clock ticks", ha='right', x=1)
    plt.ylabel("adc", ha='right', y=1)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"testdata_evt{ievt}.png")
    

def test_bool():
    """
    quick MWE for Oliver.
    the bool branches in the current file are broken in python,
    have to handle bools specially (open issue w/ h5py)
    https://github.com/h5py/h5py/pull/821
    
    the message:
    TypeError: No NumPy equivalent for TypeBitfieldID exists
    """
    hf = h5py.File("daq_testdata2.h5")
    
    bad_branches = ["inverted","muveto"]
    
    # these work
    dset = hf['/daqdata/inverted']
    ds = dset.size 
    at = dset.attrs
    for name in dset.attrs:
        print(name, dset.attrs[name])
    
    # these fail with the TypeError
    # dt = dset.dtype
    # dset = hf['/daqdata/inverted'][...] # <-- this should return a np array
    
    # try a workaround, maybe it will help identify the problem
    
    # 1. Fails, OSError: Can't read data (no appropriate function for conversion
    # bools = np.empty(dset.size, dtype=np.uint8)
    # dset.read_direct(bools)  path)
    
    # 2. Fails, same error (tried several datatypes here)
    # with dset.astype(np.uint8):
    #     bools = dset[...]
    
    hf.close()
    
    
def write_h5():
    """
    read an MJ60 raw file and convert to Oliver's format.
    this is a test function on the way to changing daq_to_raw.py in pygama.
    """
    print("hi clint")
    
    
    
    
if __name__=="__main__":
    main()