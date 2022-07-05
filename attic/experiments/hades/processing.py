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
    arg('-d2r','--d2r', action=st, help='run daq_to_raw')
    arg('-r2d','--r2d', action=st, help='run raw_to_dsp')
    arg('-st' ,'--st' , nargs=1  , help='define the scantype')
    arg('-det','--det', nargs=1  , help='define the detector') 
 
    # options
    arg('--over', action=st, help='overwrite existing files')
    
    args = par.parse_args()
    argsv = vars(par.parse_args())
    
    expDB = 'HADES.json'
   
    dg = DataGroup(config=expDB)
   
    st, det = '', '' 
    if argsv['st']:
       st = argsv['st'][0]
    if argsv['det']:
       det = argsv['det'][0]
    
    if args.d2r: d2r(dg, st, det, args.over)
    if args.r2d: r2d(dg, st, det, args.over)
    
    
def d2r(dg, st,det,overwrite=False):
    """
    run daq_to_raw on the current DataGroup
    """
    dg.load_df("HADES_DF.h5")
    qu = 'detSN=="' + det + '" & runtype == "' + st +'"' 
    keys = dg.fileDB.query(qu)
    keys.reset_index(inplace=True)
    daq_path  = dg.daq_dir
    sub_dir   = keys['daq_subdir'][0]
    daq_files = keys['daq_file']
    
    
    for b in daq_files:
       f_daq    = daq_path + '/' + sub_dir + '/' + b
       raw_file = f_daq.replace('tier0','tier1').replace('.fcio','_tier1.lh5').replace('/char_data-I','/pygama/char_data-I')
       daq_to_raw(f_daq, raw_file, config=dg.config['runDB'])


def r2d(dg, overwrite=False, nwfs=None, vrb=False):
    """
    """
    dg.load_df("HADES_DF.h5")
    qu = 'detSN=="' + det + '" & runtype == "' + st +'"'
    
    keys = dg.fileDB.query(qu)
    keys.reset_index(inplace=True)

    raw_path  = dg.daq_dir
    sub_dir   = keys['daq_subdir'][0]
    daq_files = keys['daq_file']

    for b in daq_files:
        f_daq    = daq_path + '/' + sub_dir + '/' + b
        raw_file = f_daq.replace('tier0','tier1').replace('.fcio','_tier1.lh5').replace('/char_data-I','/pygama/char_data-I')
        dsp_file = raw_file.replace('tier1','tier2')
        raw_to_dsp(raw_file,dsp_file,'r2d.json')
        

    
def raw_to_dsp(raw_file, dsp_file, json_file, buffer_len=8,group='', jout=True):
    
    lh5_store = lh5.Store()
    lh5_in = lh5_store.read_object('g000/raw',raw_file)
     
    with open(json_file) as f:
        config = json.load(f)
    paths = config["paths"]
    options = config["options"]

    channel = 'g000'
    wf_in = lh5_in["waveform"]["values"].nda
    dt = lh5_in['waveform']['dt'].nda[0] * unit_parser.parse_unit(lh5_in['waveform']['dt'].attrs['units'])

    proc = ProcessingChain(block_width=buffer_len, clock_unit=dt, verbosity=1)

    print("Now processing")

    if jout:
       proc.add_input_buffer("wf", wf_in, dtype='float32')
       proc.set_processor_list(json_file)
       lh5_out = lh5.Table(size=proc._buffer_len)
    
       for output in config["outputs"]:
          lh5_out.add_field(output, lh5.Array(proc.get_output_buffer(output)))
       
    else:
       proc.add_input_buffer("wf", wf_in, dtype='float32')
       proc.add_processor(mean_stdev, "wf[0:1000]", "bl", "bl_sig")
       proc.add_processor(np.subtract, "wf", "bl", "wf_blsub")
       proc.add_processor(pole_zero, "wf_blsub", 70*us, "wf_pz")
       proc.add_processor(trap_norm, "wf_pz", 10*us, 5*us, "wf_trap")
       proc.add_processor(asymTrapFilter, "wf_pz", 0.05*us, 2*us, 4*us, "wf_atrap")

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
    
