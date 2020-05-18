#!/usr/bin/env python3
import os
import numpy as np
import argparse
import pandas as pd
import json
import h5py
from pprint import pprint

from pygama import DataGroup
from pygama.io.daq_to_raw import daq_to_raw
from pygama.io import lh5
from pygama.dsp.units import *
from pygama.dsp.ProcessingChain import ProcessingChain
from pygama.dsp.processors import *

def main():
    doc="""
    LPGTA data processing routine. 
    You must set these environment variables:
      * $LPGTA_DATA : base data directory
      * $LEGEND_META : the legend-metadata repository
    """
    rthf = argparse.RawTextHelpFormatter
    par = argparse.ArgumentParser(description=doc, formatter_class=rthf)
    arg, st, sf = par.add_argument, 'store_true', 'store_false'
    
    # routines
    arg('--d2r', action=st, help='run daq_to_raw')
    arg('--r2d', action=st, help='run raw_to_dsp')
    
    # options
    arg('--over', action=st, help='overwrite existing files')
    
    args = par.parse_args()
    
    expDB = 'HADES.json'
   
    dg = DataGroup(config=expDB)
    

    
    if args.d2r: d2r(dg, args.over)
    if args.r2d: r2d(dg, args.over)
    
    
def d2r(dg, overwrite=False):
    """
    run daq_to_raw on the current DataGroup
    """
    dg.scan_daq_dir()
    st = 'bkg'
    keys = dg.file_keys    
    b60 = dg.file_keys.query('detSN=="I02160B" & scantype == "bkg"')
    b60.reset_index(inplace=True)
    daq_path  = dg.daq_dir
    sub_dir   = b60['daq_subdir'][0]
    daq_files = b60['daq_file']
    
    
    for b in daq_files:
       f_daq = daq_path + '/' + sub_dir + '/' + b
       lh5_dir = f_daq.replace('tier0','tier1').replace('.fcio','_tier1.lh5').replace('/char_data-I','/pygama/char_data-I')
       daq_to_raw(f_daq, lh5_dir, config=dg.config['runDB'])
       return


def r2d(dg, overwrite=False, nwfs=None, vrb=False):
    """
    """
    df_raw = dg.scan_raw_dir()
    keys = dg.file_keys
    b60 = dg.file_keys.query('detSN=="I02160B" & scantype == "bkg"')
    b60.reset_index(inplace=True)

    raw_path  = dg.raw_dir
    sub_dir   = b60['raw_subdir'][0]
    raw_files = b60['raw_file']

    for f in raw_files:
        raw_file = raw_path + '/' + sub_dir +'/' + f
        dsp_file = raw_file.replace('tier1','tier2')
        raw_to_dsp(raw_file,dsp_file,'raw_to_dsp.json')
        

    
def raw_to_dsp(raw_file, dsp_file, json_file, buffer_len=8,group=''):
    
    lh5_store = lh5.Store()
    lh5_in = lh5_store.read_object('g000/raw',raw_file)
     
    with open(json_file) as f:
        config = json.load(f)
    paths = config["paths"]
    options = config["options"]

    channel = 'g000'
    wf_in = lh5_in["waveform"]["values"].nda
    dt = lh5_in['waveform']['dt'].nda[0] * unit_parser.parse_unit(lh5_in['waveform']['dt'].attrs['units'])

    print("Now processing")
    proc = ProcessingChain(block_width=buffer_len, clock_unit=dt, verbosity=1) 
    proc.add_input_buffer("wf", wf_in, dtype='float32')
    proc.add_processor(mean_stdev, "wf[0:1000]", "bl", "bl_sig")
    proc.add_processor(np.subtract, "wf", "bl", "wf_blsub")
    proc.add_processor(pole_zero, "wf_blsub", 70*us, "wf_pz")
    proc.add_processor(trap_norm, "wf_pz", 10*us, 5*us, "wf_trap")
    proc.add_processor(asymTrapFilter, "wf_pz", 0.05*us, 2*us, 4*us, "wf_atrap")

    jout = False
    lh5_out = lh5.Table(size=proc._buffer_len)
  
    if jout:
     for output in config["outputs"]:

        lh5_out.add_field(output, lh5.Array(proc.get_output_buffer(output)))

    else:
        lh5_out.add_field("bl", lh5.Array(proc.get_output_buffer("bl"), attrs={"units":"ADC"}))
        lh5_out.add_field("bl_sig", lh5.Array(proc.get_output_buffer("bl_sig"), attrs={"units":"ADC"}))

    proc.execute()
    groupname = "/data"
    lh5_store.write_object(lh5_out, groupname, dsp_file)
 
if __name__=="__main__":
    main()
    
