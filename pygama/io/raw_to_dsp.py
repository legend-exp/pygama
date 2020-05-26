#!/usr/bin/env python3
import os
import json
import h5py
import time
import numpy as np
from collections import OrderedDict
from pprint import pprint

from pygama.dsp.ProcessingChain import ProcessingChain
from pygama.dsp.units import *
from pygama.io import lh5


def raw_to_dsp(f_raw, f_dsp, dsp_config, lh5_tables=None, verbose=False, 
               n_max=np.inf, overwrite=True, buffer_len=8):
    """
    Convert raw LH5 files with waveforms to dsp files.
    
    Uses the ProcessingChain class.  
    The list of processors is specifed via a JSON file.  
    To preserve the ordering, we read in using an OrderedDict.
    """
    t_start = time.time()
    
    if not isinstance(dsp_config, OrderedDict):
        print('Error, dsp_config must be an OrderedDict')
        exit()
        
    raw_store = lh5.Store()
    
    # if no group is specified, assume we want to decode every table in the file
    if lh5_tables is None:
        lh5_tables = raw_store.ls(f_raw)

    # set up DSP for each table
    chains = []
    for tb in lh5_tables:
        print('Processing table: ', tb)
        
        # load primary table
        data_raw = raw_store.read_object(tb, f_raw, start_row=0, n_rows=n_max)
        
        # load waveform info
        if "waveform" not in data_raw.keys():
            print(f"waveform data not found in table: {tb}.  skipping ...")
            continue
        wf_in = data_raw["waveform"]["values"].nda
        wf_units = data_raw['waveform']['dt'].attrs['units']
        dt = data_raw['waveform']['dt'].nda[0] * unit_parser.parse_unit(wf_units)
        
        # set up DSP for this table (JIT-compiles functions the first time)
        pc = ProcessingChain(block_width=buffer_len, clock_unit=dt, verbosity=0)
        pc.add_input_buffer('wf', wf_in, dtype='float32')
        pc.set_processor_list(dsp_config)

        # set up LH5 output table 
        tb_out = lh5.Table(size = pc._buffer_len)
        cols = dsp_config['outputs']
        # cols = pc.get_column_names() # TODO: should add this method
        for col in cols:
            lh5_arr = lh5.Array(pc.get_output_buffer(col),
                                attrs={'units':dsp_config['outputs'][col]})
            tb_out.add_field(col, lh5_arr)
            
        chains.append((tb, tb_out, pc))


    # run DSP.  TODO: parallelize this
    print('Writing to output file:', f_dsp)
    for tb, tb_out, pc in chains:
        print(f'Processing table: {tb} ...')
        # pc.execute()
        # print(f'Done.  Writing to file ...')
        # raw_store.write_object(tb_out, tb, f_dsp)
    
    t_elap = (time.time() - t_start) / 60
    print(f'Done processing.  Time elapsed: {t_elap:.2f} min.')
    

if __name__=="__main__":
    example()
