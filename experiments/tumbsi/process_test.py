#!/usr/bin/env python3
import sys, os, io
import json
import argparse
import numpy as np
import pandas as pd
from pprint import pprint
from pygama import DataSet

def main(argv):
  """
  Uses pygama's amazing DataSet class to process runs for different
  data sets, with arbitrary configuration options defined in a JSON file.
  C. Wiseman, 2019/04/09
  """
  run_db = './runDB.json'

  # -- parse args --
  par = argparse.ArgumentParser(description="test data processing suite")
  arg, st, sf = par.add_argument, "store_true", "store_false"
  arg("-ds", nargs='*', action="store", help="load runs for a DS")
  arg("-r", "--run", nargs=1, help="load a single run")
  arg("-t0", "--daq_to_raw", action=st, help="run ProcessRaw on list")
  arg("-t1", "--raw_to_dsp", action=st, help="run RunDSP on list")
  arg("-t", "--test", action=st, help="test mode, don't run")
  arg("-n", "--nevt", nargs='?', default=np.inf, help="limit max num events")
  arg("-i", "--ioff", nargs='?', default=0, help="start at index [i]")
  arg("-v", "--verbose", action=st, help="set verbose output")
  arg("-o", "--ovr", action=st, help="overwrite existing files")
  arg("-m", "--nomp", action=sf, help="don't use multiprocessing")
  args = vars(par.parse_args())

  
  # -- declare the DataSet --
  if args["ds"]:
    ds_lo = int(args["ds"][0])
    try:
      ds_hi = int(args["ds"][1])
    except:
      ds_hi = None
    ds = DataSet(ds_lo, ds_hi, md=run_db, v=args["verbose"])
  
    if args["run"]:
      ds = DataSet(run=int(args["run"][0]), md=run_db, v=args["verbose"])
    
    # -- start processing --
    if args["daq_to_raw"]:
      daq_to_raw(ds, args["ovr"], args["nevt"], args["verbose"], args["test"])
    
    if args["raw_to_dsp"]:
      raw_to_dsp(ds, args["ovr"], args["nevt"], args["ioff"], args["nomp"], args["verbose"],
            args["test"])



def daq_to_raw(ds,overwrite=False, nevt=np.inf, v=False, test=False):
    """
    Run ProcessRaw on a set of runs.
    [raw file] ---> [t1_run{}.h5] (tier 1 file: basic info & waveforms)
    """
    from pygama.io.daq_to_raw import ProcessRaw
    
    for run in ds.runs:
      print(ds.raw_dir)
      t0_file = ds.paths[run]["t0_path"]
      t1_file = ds.paths[run]["t1_path"]
      if t1_file is not None and overwrite is False:
        continue
        
      print("In/Out files:",t0_file,t1_file)
      conf = ds.paths[run]["build_opt"]
      opts = ds.config["build_options"][conf]["daq_to_raw_options"]
      
      if test:
        print("test mode (dry run), processing Tier 0 file:", t0_file)
        continue
      
      if nevt != np.inf:
        nevt = int(nevt)
        
      ProcessRaw(
                   t0_file,
                   run,
                   verbose=v,
                   output_dir=ds.tier_dir,
                   overwrite=overwrite,
                   n_max=nevt,
                   settings=opts)

def raw_to_dsp(ds,
          overwrite=False,
          nevt=None,
          ioff=None,
          multiproc=None,
          verbose=False,
          test=False):
    """
    Run RunDSP on a set of runs.
    [t1_run{}.h5] ---> [t2_run{}.h5]  (tier 2 file: DSP results, no waveforms)

    Can declare the processor list via:
    - json configuration file (recommended)
    - Intercom(default_list=True)
    - manually add with Intercom::add
    """
    from pygama.dsp.dsp_base import Intercom
    from pygama.io.raw_to_dsp import RunDSP

    for run in ds.runs:

        t1_file = ds.paths[run]["t1_path"]
        t2_file = ds.paths[run]["t2_path"]
        if t2_file is not None and overwrite is False:
            continue

        if test:
            print("test mode (dry run), processing Tier 1 file:", t1_file)
            continue

        conf = ds.paths[run]["build_opt"]
        proc_list = ds.config["build_options"][conf]["raw_to_dsp_options"]
        proc = Intercom(proc_list)

        RunDSP(
            t1_file,
            proc,
            output_dir=ds.tier_dir,
            overwrite=overwrite,
            verbose=verbose,
            multiprocess=multiproc,
            nevt=nevt,
            ioff=ioff,
            chunk=ds.config["chunksize"])


if __name__ == "__main__":
    main(sys.argv[1:])
