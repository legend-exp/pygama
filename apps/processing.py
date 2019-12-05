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
    run_db = './runDB.json'
    # -- parse args --
    par = argparse.ArgumentParser(description="data processing suite for MJ60")
    arg, st, sf = par.add_argument, "store_true", "store_false"
    arg("-ds", nargs='*', action="store", help="load runs for a DS")
    arg("-r", "--run", nargs=1, help="load a single run")
    arg("-t0", "--tier0", action=st, help="run ProcessTier0 on list")
    arg("-t1", "--tier1", action=st, help="run ProcessTier1 on list")
    arg("-t", "--test", action=st, help="test mode, don't run")
    arg("-n", "--nevt", nargs='?', default=np.inf, help="limit max num events")
    arg("-i", "--ioff", nargs='?', default=0, help="start at index [i]")
    arg("-v", "--verbose", action=st, help="set verbose output")
    arg("-o", "--ovr", action=st, help="overwrite existing files")
    arg("-m", "--nomp", action=sf, help="don't use multiprocessing")
    args = vars(par.parse_args())

    ds = pu.get_dataset_from_cmdline(args, "runDB.json", "calDB.json")

    # -- start processing --
    if args["tier0"]:
        tier0(ds, args["ovr"], args["nevt"], args["verbose"], args["test"])

    if args["tier1"]:
        tier1(ds, args["ovr"], args["nevt"], args["ioff"], args["nomp"], args["verbose"],
              args["test"])


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
            print("file exists, overwrite flag isn't set.  continuing ...")
            continue

        conf = ds.paths[run]["build_opt"]
        opts = ds.config["build_options"][conf]["tier0_options"]

        if test:
            print("test mode (dry run), processing Tier 0 file:", t0_file)
            print("writing to:", t1_file)
            continue

        ProcessTier0(
            t0_file,
            run,
            verbose=v,
            output_dir=ds.tier1_dir,
            overwrite=overwrite,
            n_max=nevt,
            settings=opts)


def tier1(ds,
          overwrite=False,
          nevt=None,
          ioff=None,
          multiproc=True,
          verbose=False,
          test=False):
    """
    Run ProcessTier1 on a set of runs.
    [t1_run{}.h5] ---> [t2_run{}.h5]  (tier 2 file: DSP results, no waveforms)

    Can declare the processor list via:
    - json configuration file (recommended)
    - Intercom(default_list=True)
    - manually add with Intercom::add
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
        proc_list = ds.config["build_options"][conf]["tier1_options"]
        proc = Intercom(proc_list)

        ProcessTier1(
            t1_file,
            proc,
            output_dir=ds.tier2_dir,
            overwrite=overwrite,
            verbose=verbose,
            multiprocess=multiproc,
            nevt=nevt,
            ioff=ioff,
            chunk=ds.config["chunksize"])


if __name__ == "__main__":
    main(sys.argv[1:])
