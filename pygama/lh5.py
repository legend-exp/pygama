#!/usr/bin/env python3
import os
import time
import h5py
import numpy as np
import pandas as pd
from pprint import pprint


def dir_lh5(in_file):
    """
    access an lh5 file and pretty print the structure.
    TODO: add verbosity option
    NOTE: calling hf.visit or hf.visititems iterates over groups automatically, 
    instead of an explicit loop.  the h5py book tells you to do it this way.
    """
    hf = h5py.File(in_file)
    
    def print_groups(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"GROUP /{name}")
            indent = "  "
        if isinstance(obj, h5py.Dataset):
            print("  DATASET", obj.shape, obj.name)
            indent = "    "
        for att, val in obj.attrs.items():
            print(f"{indent}ATTRIBUTE {att}:", val)
        print(" ")

    hf.visititems(print_groups)

                            
def read_table(table_name, hf, df_fmt, get_wfs=False):
    """
    internal function for read_lh5
    
    Wisdom from Jeff Reback about doing this efficiently:
    "you can simply create an empty frame with an index and columns, then assign
    ndarrays - these won't copy if you assign all of a particular dtype at once. 
    you could create these with np.empty if you wish."
    """
    dfs = []
    for dt, block in df_fmt.groupby("dtype"):
        
        ncols = len(block)
        nrows = block["shape"].unique()
        if len(nrows) > 1:
            print("Error, columns are different lengths")
            exit()
        nrows = nrows[0][0]

        # preallocate a block for this dtype
        np_block = np.empty((nrows, ncols), dtype=dt)
        
        for i, col in enumerate(block["name"]):
            
            ds = hf[f"{table_name}/{col}"] # reference the ds w/o reading
            np_block[:,i] = ds[...]   # read into memory
            
        dfs.append(pd.DataFrame(np_block, columns=block["name"]))

    # concat final DF after grouping dtypes and avoiding copies
    return pd.concat(dfs, axis=1, copy=False)
        

def read_waveforms(table_name, hf, df_fmt, ilo=0, ihi=None):
    """
    efficiently decompress waveforms in an LH5 file into a rectangular ndarray,
    which can be concatenated into the main 'tier 1' waveform dataframe
    """
    ds_clen = hf[f"{table_name}/values/cumulative_length"]
    ds_flat = hf[f"{table_name}/values/flattened_data"]
    
    nwf_tot = ds_clen.shape[0]
    nval_tot = ds_flat.shape[0]
    
    if ihi is None:
        ihi = nwf_tot
    nwfs = ihi - ilo + 1 # inclusive
    
    # find indexes of raw values to read in
    clo = ds_clen[ilo]
    chi = int(ds_clen[ihi+1] if ihi != nwf_tot else nval_tot)
    
    # read raw values and the set of first indexes into memory
    wf_vals = ds_flat[clo:chi] 
    wf_idxs = ds_clen[ilo:ihi+1] if ihi!= nwf_tot else ds_clen[ilo:]

    # split the flattened data by our set of indexes
    loc_idxs = (wf_idxs - wf_idxs[0])[1:] # ignore the 0 value
    wf_list = np.array_split(wf_vals, loc_idxs)
    
    # TODO: here's where I would decompress waveforms using a fast C++ function
    
    # now that all wfs are same size, fill and return an ndarray
    return np.vstack(wf_list)


def read_lh5(in_file, header=False, cols=None, ilo=None, ihi=None, ds=None):
    """
    Convert lh5 to pandas DF, loading it into memory with minimal copying.  
    
    This function should be very general and include many keyword arguments, 
    just like the way pandas.read_hdf works.  
    
    There are three standard use cases we should support:
    Usage 1:
    >>> from pygama import read_lh5
    >>> df = read_lh5(file, *args)

    Usage 2:
    >>> import pygama as pg 
    >>> df = pg.read_lh5(file, *args)

    TODO: Usage 3: call it with additional metadata from DataSet
    >>> from pygama import DataSet
    >>> ds = DataSet(...)
    >>> df = ds.read_lh5(file, *args)
    """
    if ".lh5" not in in_file:
        print("Error, unknown file:", in_file)
        exit()
    
    tables = {}
    def lookup_tables(name, obj):
        """
        for each table in the LH5 file, save column, size, and dtype 
        so that we can preallocate numpy arrays before reading the data.
        """ 
        if isinstance(obj, h5py.Group):
            for att, val in obj.attrs.items():
                if att == "datatype" and "table{" in val:
                    cols = [] 
                    table_dfn = val[6:-1].split(",")
                    for col in table_dfn:
                        ds_path = f"{name}/{col}"
                        ds = hf[ds_path]
                        s, d = None, None
                        if isinstance(ds, h5py.Dataset):
                            s, d = ds.shape, ds.dtype
                        cols.append((col, s, d))
                    tables[name] = cols

    # open the file in context manager to avoid weird crashes 
    t_start = time.time()
    with h5py.File(os.path.expanduser(in_file)) as hf:
        
        # find all tables in the file
        hf.visititems(lookup_tables)
        # pprint(tables)
        
        # read single-valued tables (ignoring "None" columns)
        # tname = "daqdata"
        # df_fmt = pd.DataFrame(tables[tname], columns=["name","shape","dtype"])
        # df = read_table(tname, hf, df_fmt)
        # print(df_fmt)
        # print(df)

        # read waveform-valued tables (un-flattening & decompressing wfs)
        wname = "daqdata/waveform"
        df_wfs = pd.DataFrame(tables[wname], columns=["name","shape","dtype"])
        
        df = read_table(wname, hf, df_wfs, get_wfs=True)
        
        # df = read_waveforms(wname, hf, df_wfs)
        print(df_wfs)
        print(df)
    
    # t_elapsed = time.time() - t_start
    # print("elapsed: {t_elapsed:.4f} sec")
    
    # return df


if __name__=="__main__":
    """
    debug functions
    """
    f_lh5 = "/Users/wisecg/Data/L200/tier1/t1_run0.lh5"
    dir_lh5(f_lh5)
    df = read_lh5(f_lh5)
    print(df)