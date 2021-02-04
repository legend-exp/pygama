#!/usr/bin/env python3
import h5py
import numpy as np
import pandas as pd
from pprint import pprint

from tqdm import tqdm
tqdm.pandas()

import matplotlib.pyplot as plt
plt.style.use('../../pygama/clint.mpl')

from pygama import DataGroup
import pygama.io.lh5 as lh5


def main():
    """
    sandbox code for exploring LH5 file groups, loading channel maps, etc.
    """
    test_datagroup()
    

def test_datagroup():
    """
    current columns:
    ['unique_key', 'run', 'label', 'YYYYmmdd', 'hhmmss', 'rtp', 'daq_dir',
     'daq_file', 'cmap', 'runtype', 'raw_file', 'raw_path', 'dsp_file',
     'dsp_path', 'hit_file', 'hit_path', 'daq_size_GB', 'proc_group']
    """
    dg = DataGroup('LPGTA.json', load=True)
    query = "run==30 and rtp == 'calib' and proc_group==35"
    dg.fileDB.query(query, inplace=True)
    # dg.fileDB = dg.fileDB[-1:]
    # print(dg.fileDB.columns)

    # show what has been selected
    view_cols = ['run', 'label', 'YYYYmmdd', 'hhmmss', 'rtp', 'cmap', 'runtype',
                 'daq_size_GB', 'proc_group']
    # print(dg.fileDB[view_cols].to_string())
    
    raw_path, raw_file = dg.fileDB[['raw_path','raw_file']].iloc[0]
    f_raw = f'{dg.lh5_dir}/{raw_path}/{raw_file}'
    
    if "sysn" in f_raw:
        tmp = {'sysn' : 'geds'} # hack for lpgta
        f_raw = f_raw.format_map(tmp)
        # f_dsp = f_dsp.format_map(tmp)
    
    # check_lh5_groups(f_raw)
    # load_raw_data_example(f_raw)
    check_timestamps(f_raw)
    

def check_lh5_groups(f_lh5):
    """
    useful but verbose.
    open an LH5 file store and identify all groups, datatypes, etc.
    """
    def print_attrs(name, obj):
        print(name) # show group name only
        # show attributes (lh5 datatypes)
        for key, val in obj.attrs.items():
            print(f"    {key}: {val}")
    f = h5py.File(f_lh5, 'r')
    f.visititems(print_attrs)

    
def load_raw_data_example(f_raw):
    """
    make a plot of the timestamps in a particular channel.
    instead of accessing just the timestamp column, this is an example
    of accessing the entire raw file (including waveforms) with LH5.
    """
    sto = lh5.Store()
    
    tb_name = 'g024/raw'
    
    n_rows = 100 # np.inf to read all
    
    # method 1: call load_nda to pull out only timestamp column (fast)
    # par_data = lh5.load_nda([f_raw], ['timestamp'], tb_name)
    # pprint(par_data)
    # print(par_data['timestamp'].shape)
    # exit()
    
    # method 2: read all data, just to give a longer example of what we can access
    # TODO: include an example of slicing/selecting rows with np.where
    
    # read non-wf cols (lh5 Arrays)
    data_raw, n_tot = sto.read_object(tb_name, f_raw, n_rows=n_rows)

    # declare output table (must specify n_rows for size)
    tb_raw = lh5.Table(size=n_tot)

    for col in data_raw.keys():
        if col in ['waveform','tracelist']: continue
        # copy all values
        newcol = lh5.Array(data_raw[col].nda, attrs=data_raw[col].attrs)
        # copy a selection (using np.where)
        # newcol = lh5.Array(data_raw[col].nda[idx], attrs=data_raw[col].attrs)
        tb_raw.add_field(col, newcol)
        
    df_raw = tb_raw.get_dataframe()
    print(df_raw)
    
    # load waveform column (nested LH5 Table)
    data_wfs, n_tot = sto.read_object(tb_name+'/waveform', f_raw, n_rows=n_rows)
    tb_wfs = lh5.Table(size=n_tot)
    
    for col in data_wfs.keys():
        attrs = data_wfs[col].attrs
        if isinstance(data_wfs[col], lh5.ArrayOfEqualSizedArrays):
            # idk why i can't put the filtered array into the constructor
            aoesa = lh5.ArrayOfEqualSizedArrays(attrs=attrs, dims=[1,1])
            aoesa.nda = data_wfs[col].nda
            # aoesa.nda = data_wfs[col].nda[idx] # with np.where selection
            newcol = aoesa
        else:
            newcol = lh5.Array(data_wfs[col].nda, attrs=attrs)
            # newcol = lh5.Array(data_wfs[col].nda[idx], attrs=attrs) # selection
        tb_wfs.add_field(col, newcol)
            
    tb_wfs.add_field('waveform', newcol)
    tb_wfs.attrs = data_raw.attrs
    
    # can write to file, to read back in for DSP, etc.
    # sto.write_object(tb_raw, grp_data, f_peak)
    
    print(tb_wfs)
    print(tb_wfs['waveform'].shape)
    
    
def check_timestamps(f_raw):
    """
    fc daq timestamps are in seconds, from beginning of file:
    https://github.com/legend-exp/pygama/blob/master/pygama/io/fcdaq.py#L27
    """
    ts = lh5.load_nda([f_raw], ['timestamp'], 'g024/raw')['timestamp']
    
    print(ts)
    print(ts.shape)
    print(f'first: {ts[0]}  {min(ts)}  last: {ts[-1]}  {max(ts)}')
    
    rt = ts[-1] / 60 # runtime in min
    
    plt.plot(np.arange(len(ts)), ts, '.b', label=f'runtime: {rt:.1f} min')
    plt.xlabel('entry', ha='right', x=1)
    plt.ylabel('timestamp', ha='right', y=1)
    plt.legend()
    plt.savefig('./plots/ts_check.png', dpi=100)
    
    
if __name__=='__main__':
    main()
    