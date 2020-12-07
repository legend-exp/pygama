#!/usr/bin/env python3
import h5py
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
    dg.file_keys.query(query, inplace=True)
    # dg.file_keys = dg.file_keys[-1:]
    # print(dg.file_keys.columns)

    # show what has been selected
    view_cols = ['run', 'label', 'YYYYmmdd', 'hhmmss', 'rtp', 'cmap', 'runtype',
                 'daq_size_GB', 'proc_group']
    # print(dg.file_keys[view_cols].to_string())
    
    raw_path, raw_file = dg.file_keys[['raw_path','raw_file']].iloc[0]
    f_raw = f'{dg.lh5_dir}/{raw_path}/{raw_file}'
    
    if "sysn" in f_raw:
        tmp = {'sysn' : 'geds'} # hack for lpgta
        f_raw = f_raw.format_map(tmp)
        # f_dsp = f_dsp.format_map(tmp)
    
    check_lh5_groups(f_raw)
    

def check_lh5_groups(f_lh5):
    """
    open an LH5 file store and identify all groups, datatypes, etc.
    """
    # quick dump of all group info with h5py (useful but verbose)
    def print_attrs(name, obj):
        print(name) # show group name only
        # show attributes (lh5 datatypes)
        for key, val in obj.attrs.items():
            print(f"    {key}: {val}")
    f = h5py.File(f_lh5, 'r')
    f.visititems(print_attrs)
    
    
    
    
    
    # # sto = lh5.Store()
    # # groups = sto.ls(f_lh5, '')
    # # 
    # # for grp in groups:
    # # 
    # #     subgrp = sto.ls(f_lh5, grp+'/')
    # 
    # 
    #     # # read non-wf cols (lh5 Arrays)
    #     # data_raw = sto.read_object(tb_raw, f_raw, n_rows=n_rows)
    #     # for col in data_raw.keys():
    #     #     if col=='waveform': continue
    #     #     newcol = lh5.Array(data_raw[col].nda[idx], attrs=data_raw[col].attrs)
    #     #     tb_data.add_field(col, newcol)
    #     # 
    #     # # handle waveform column (lh5 Table)
    #     # data_wfs = sto.read_object(tb_raw+'/waveform', f_raw, n_rows=n_rows)
    #     # for col in data_wfs.keys():
    #     #     attrs = data_wfs[col].attrs
    #     #     if isinstance(data_wfs[col], lh5.ArrayOfEqualSizedArrays):
    #     #         # idk why i can't put the filtered array into the constructor
    #     #         aoesa = lh5.ArrayOfEqualSizedArrays(attrs=attrs, dims=[1,1])
    #     #         aoesa.nda = data_wfs[col].nda[idx]
    #     #         newcol = aoesa
    #     #     else:
    #     #         newcol = lh5.Array(data_wfs[col].nda[idx], attrs=attrs)
    #     #     wf_tb_data.add_field(col, newcol)
    #     # tb_data.add_field('waveform', wf_tb_data)
    #     # tb_data.attrs = data_raw.attrs
    #     # sto.write_object(tb_data, grp_data, f_peak)
    # 
    #     print(grp)
    #     print(subgrp)
    # 
    #     exit()
    
    
if __name__=='__main__':
    main()
    