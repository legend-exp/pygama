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
    Uses pygama's amazing DataSet class to process runs
    for different data sets and arbitrary configuration options
    defined in a JSON file.
    """
    metad = './runDB.json'

    # -- parse args --
    par = argparse.ArgumentParser(description="pygama data processing suite for MJ60")
    arg = par.add_argument
    arg("-ds", nargs='*', action="store", help="load runs for a DS")
    arg("-r","--run", nargs=1, help="load a single run")
    arg("-t0","--tier0", action="store_true", help="run ProcessTier0 on run list")
    arg("-t1","--tier1", action="store_true", help="run ProcessTier1 on run list")
    arg("-t","--test", action="store_true", help="test mode, don't run pygama routines")
    arg("-n","--nevt", nargs='?', default=np.inf, help="limit max events per file")
    arg("-v","--verbose", action="store_true", help="set verbose output")
    arg("-o","--overwrite", action="store_true", help="overwrite existing files")
    arg("-m","--nomp", action="store_false", help="don't use multiprocessing")
    args = vars(par.parse_args())

    # -- declare the DataSet --
    if args["ds"]:
        ds_lo = int(args["ds"][0])
        try:
            ds_hi = int(args["ds"][1])
        except:
            ds_hi = None
        ds = DataSet(ds_lo, ds_hi, md=metad, v=args["verbose"])

    if args["run"]:
        ds = DataSet(run=int(args["run"][0]), md=metad, v=args["verbose"])

    # -- start processing --
    if args["tier0"]:
        tier0(ds, args["overwrite"], args["nevt"], args["verbose"], args["test"])

    if args["tier1"]:
        tier1(ds, args["overwrite"], args["nevt"], args["nomp"],
              args["verbose"], args["test"])


def tier0(ds, overwrite=False, nevt=np.inf, v=False, test=False):
    """
    Run ProcessTier0 on a set of runs.
    [raw file] ---> [t1_run{}.h5] (tier 1 file: basic info & waveforms)
    """
    from pygama.io.tier0 import ProcessTier0

    for run in ds.runs:

        t0_file = ds.paths[run]["t0_path"]
        t1_file = ds.paths[run]["t1_path"]
        if t1_file is not None and overwrite is False:
            continue

        conf = ds.paths[run]["build_opt"]
        opts = ds.runDB["build_options"][conf]["tier0_options"]

        if test:
            print("test mode (dry run), processing Tier 0 file:", t0_file)
            continue

        ProcessTier0(t0_file,
                     verbose = v,
                     output_dir = ds.tier_dir,
                     overwrite = overwrite,
                     n_max = nevt,
                     settings = opts)


def tier1(ds, overwrite=False, nevt=None, multiproc=True, verbose=False, test=False):
    """
    Run ProcessTier1 on a set of runs.
    Can declare the processor list via:
        - json configuration file
        - Intercom(default_list=True)
        - manually add with Intercom::add
    [t1_run{}.h5] ---> [t2_run{}.h5] (tier 2 file: DSP results, no waveforms)
    """
    from pygama.dsp.base import Intercom
    from pygama.io.tier1 import ProcessTier1

    for run in ds.runs:

        t1_file = ds.paths[run]["t1_path"]
        t2_file = ds.paths[run]["t2_path"]
        if t2_file is not None and overwrite is False:
            continue

        if test:
            print("test mode (dry run), processing Tier 1 file:", t1_file)
            continue

        conf = ds.paths[run]["build_opt"]
        proc_list = ds.runDB["build_options"][conf]["tier1_options"]
        proc = Intercom(proc_list)

        ProcessTier1(t1_file, proc,
                     output_dir = ds.tier_dir,
                     overwrite = overwrite,
                     verbose = verbose,
                     multiprocess = multiproc,
                     nevt = nevt,
                     chunk = ds.runDB["chunksize"])


if __name__=="__main__":
    main(sys.argv[1:])
