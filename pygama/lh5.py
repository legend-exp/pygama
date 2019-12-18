#!/usr/bin/env python3
import os
import time
import h5py
import numpy as np
import pandas as pd
from pprint import pprint


def get_lh5_header(in_file, verbose=False):
    """
    Access an lh5 file, pretty print the structure, and return a useful header.
    
    Usage:
    >>> import pygama as pg 
    >>> pg.get_lh5_header(file, verbose=True) # pretty print the file structure
    >>> hdr = pg.get_lh5_header(file, *args)  # gives a dict of DataFrames

    A handy reference page for doing efficient iteration over h5py groups:
    https://stackoverflow.com/questions/45562169/traverse-hdf5-file-tree-and-continue-after-return
    """
    hf = h5py.File(in_file)

    # pretty print the raw structure, with all attributes
    if verbose:
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
        hf.visititems(print_groups) # accesses __call__
    
    # find each LH5 "Table" contained in the file, and create a DataFrame header
    tables = {}
    for g_top in hf.keys():
        
        h5group = hf[f"/{g_top}"]
        attrs = {att:val for att, val in h5group.attrs.items()}
        
        # LH5 table condition
        if "datatype" in attrs.keys() and "table{" in attrs["datatype"]:
            
            # call our nice iterator at this group level
            table = {g_top:[]}
            for (path, name, size, dtype, units, spec) in get_datasets(h5group):
                table[g_top].append((name, size, dtype, units, spec))
            
            hdr = pd.DataFrame(table[g_top], columns=['name','size','dtype',
                                                      'units','spec'])
            
            # fix waveform datatype to match flattened_data
            if 'waveform' in hdr['name'].values:
                wf_dt = h5group['waveform/values/flattened_data'].dtype
                hdr.loc[hdr['name'] == 'waveform', ['dtype']] = wf_dt
            
            tables[g_top] = hdr

    return tables


def get_datasets(h5group, prefix=''):
    """
    this is an iterator that lets you recursively build a list of all (names, dtypes, lengths) of each HDF5 dataset that is a member of the given group.
    """
    for key in h5group.keys():
        h5obj = h5group[key]
        path = '{}/{}'.format(prefix, key)
        attrs = {att:val for att, val in h5obj.attrs.items()}

        if isinstance(h5obj, h5py.Dataset): 
            
            # get metadata
            units = attrs["units"] if 'units' in attrs else None
            spec = attrs["datatype"] if 'datatype' in attrs else None
        
            # special handling for the nested waveform dataset
            if "waveform/values/cumulative_length" in path:
                nwfs = h5obj.shape[0]
                
                # must fix datatype AFTER this initial iteration
                yield (path, "waveform", nwfs, None, units, spec) 
            elif "waveform" in path:
                pass
            
            # handle normal 'array<1>{real}' datasets
            else:
                yield (path, key, h5obj.shape[0], h5obj.dtype, units, spec) 
            
        # test for group (go down)
        elif isinstance(h5obj, h5py.Group): 
            yield from get_datasets(h5obj, path)


def read_lh5(in_file, key=None, cols=None, ilo=0, ihi=None):
    """
    Convert on-disk LH5 to pandas DataFrame in memory, efficiently!
    
    This function should be very general and include many keyword arguments, 
    just like the way pandas.read_hdf works.  It will have special handling for
    "waveform"-valued tables.
    
    If key is None, return a df from the first Table we find in the file.
    
    Usage:
        import pygama as pg 
        df = pg.read_lh5(file, *args)

    TODO: call this with additional metadata from DataSet, which could allow 
    the header structure to be more generalized
        from pygama import DataSet
        ds = DataSet(...)
        hdr = ds.get_header(file, *args)
        df = ds.read_lh5(file, *args)
    """
    if ".lh5" not in in_file:
        print("Error, unknown file:", in_file)
        exit()
    
    # open the file in context manager to avoid weird crashes 
    t_start = time.time()
    with h5py.File(os.path.expanduser(in_file)) as hf:
        
        header = get_lh5_header(f_lh5, verbose=False)

        # pick off first table by default, or let the user specify the name
        table = list(header.keys())[0] if key is None else key
        df_hdr = header[table]    
        
        # this function reads the Table into memory
        df = read_table(table, hf, df_hdr, ilo, ihi)

    # t_elapsed = time.time() - t_start
    # print("elapsed: {t_elapsed:.4f} sec")
    
    return df

                            
def read_table(table_name, hf, df_fmt, ilo, ihi):
    """
    internal function for read_lh5.
    
    We want to read from the LH5 file, using h5py operations, with the absolute
    minimum number of allocations and copy operations possible.
    
    Wisdom from Jeff Reback about doing this efficiently:
    "you can simply create an empty frame with an index and columns, then assign
    ndarrays - these won't copy IF you assign all of a particular dtype at once. 
    you could create these with np.empty if you wish."
    
    So this is fast because it's mirroring what BlockManager does internally.
    """
    dfs = []
    for dt, block in df_fmt.groupby("dtype"):
        
        # check if this dtype contains waveform data
        if 'waveform' in block['name'].values:
            wf_group = f"/{table_name}/waveform"
            wf_block = read_waveforms(wf_group, hf, df_fmt, ilo, ihi)
            wf_rows, wf_cols = wf_block.shape
            nrows = wf_rows
            
            # get number of additional columns
            new_cols = [c for c in list(block["name"].values) if c != 'waveform']
            newcols = len(new_cols)
            
            # allocate the full numpy array for this dtype
            np_block = np.empty((nrows, newcols + wf_cols), dtype=dt)
            np_block[:, newcols:] = wf_block
            
            cols = []
            for i, col in enumerate(new_cols):
                ds = hf[f"{table_name}/{col}"] 
                
                if ihi is None:
                    ihi = ds.shape[0]
                nwfs = ihi - ilo + 1 # inclusive
                
                np_block[:, i] = ds[ilo:ihi]
                cols.append(col)
            cols.extend(np.arange(wf_cols))    

            dfs.append(pd.DataFrame(np_block, columns=cols))
        
        # read normal 'array<1>{real}' columns
        else:
            ncols = len(block)
            nrows = block["size"].unique()
            if len(nrows) > 1:
                print('Error, columns are different lengths')
                exit()
            nrows = nrows[0]
            np_block = np.empty((nrows, ncols), dtype=dt)
            
            for i, col in enumerate(block["name"]):
                ds = hf[f"{table_name}/{col}"]
                np_block[:,i] = ds[...]
        
            dfs.append(pd.DataFrame(np_block, columns=block["name"]))    
        
    # concat final DF after grouping dtypes and avoiding copies
    return pd.concat(dfs, axis=1, copy=False)
        

def read_waveforms(table_name, hf, df_fmt, ilo=0, ihi=None):
    """
    internal function for read_lh5
    
    efficiently decompress waveforms in an LH5 file into a rectangular ndarray,
    which can be concatenated into the main 'tier 1' waveform dataframe
    """
    # assume LH5 structure
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


if __name__=="__main__":
    """
    debug functions
    """
    f_lh5 = "/Users/wisecg/Data/L200/tier1/t1_run0.lh5"
    
    # check the headers only
    headers = get_lh5_header(f_lh5, verbose=False)
    for table, df_hdr in headers.items():
        print("TABLE", table)
        print(df_hdr)
    
    # read data into memory
    df = read_lh5(f_lh5)
    # df = read_lh5(f_lh5, ilo=1) # broken, messes up dtypes
    print(df.head(10))