#!/usr/bin/env python3
import os, json
import pandas as pd
from pprint import pprint

class DataSet:
    """
    this is based on GATDataSet.
    can initialize with data sets, a single run, or lists of runs.
    can also load a JSON file to get a dict of metadata.
    """
    def __init__(self, ds_lo=None, ds_hi=None, run=None, runlist=None,
                 opt=None, v=False, md=None, raw_dir=None, tier_dir=None):

        # load metadata and set paths to data folders
        self.runDB = None
        if md is not None:
            self.load_metadata(md)
        try:
            self.raw_dir = self.runDB["raw_dir"]
            self.tier_dir = self.runDB["tier_dir"]
        except:
            self.raw_dir = raw_dir
            self.tier_dir = tier_dir

        # create the internal list of run numbers
        self.runs = []
        if ds_lo is not None:
            self.runs.extend(self.get_runs(ds_lo, ds_hi, v))
        if run is not None:
            self.runs.append(run)
        if runlist is not None:
            self.runs.extend(runlist)
        if opt == "-all":
            self.runs.extend(self.get_runs(verbose=v))

        # filenames for every run
        self.paths = self.get_paths(self.runs, v)

        # could store concatenated dfs here, like a TChain
        self.df = None

    def load_metadata(self, fname):
        with open(fname) as f:
            self.runDB = json.load(f)

    def add_run(self, runs):
        """
        can add single run numbers, or a list
        """
        if isinstance(runs, int):
            self.runs.append(runs)
        if isinstance(runs, list):
            self.runs.extend(runs)

    def get_runs(self, ds_lo=None, ds_hi=None, verbose=False):
        """
        using the runDB,
        create a list of data sets to process,
        then return a list of the included run numbers
        """
        if self.runDB is None:
            print("Error, runDB not set.")
            return []

        ds_list = []

        # load all data
        if ds_lo is None and ds_hi is None:
            ds_list.extend([d for d in self.runDB["ds"] if d != "note"])

        # load single ds
        elif ds_hi is None:
            ds_list.append(ds_lo)

        # load ds range
        else:
            ds_list.extend([str(d) for d in range(ds_lo, ds_hi+1)])

        run_list = []
        for ds in ds_list:
            tmp = self.runDB["ds"][str(ds)][0].split(",")
            r1 = int(tmp[0])
            r2 = int(tmp[1]) if len(tmp)>1 else None
            if r2 is None:
                run_list.append(r1)
            else:
                run_list.extend([r for r in range(r1, r2+1)]) # inclusive

        if verbose:
            print("Data Sets:",ds_list)
            print("Runs:",run_list)

        return run_list

    def get_paths(self, runs, verbose=False):
        """
        collect path info and flag nonexistent files.
        does a directory search with os.walk, which is faster than iglob
        https://stackoverflow.com/questions/1724693/find-a-file-in-python
        """
        run_dict = {r:{} for r in runs}

        for p, d, files in os.walk(self.raw_dir):
            for f in files:
                if any("Run{}".format(r) in f for r in runs):
                    run = int(f.split("Run")[-1])
                    run_dict[run]["t0_path"] = "{}/{}".format(p,f)

        for p, d, files in os.walk(self.tier_dir):
            for f in files:
                if any("t1_run{}.h5".format(r) in f for r in runs):
                    run = int(f.split("run")[-1].split(".h5")[0])
                    run_dict[run]["t1_path"] = "{}/{}".format(p,f)

                if any("t2_run{}.h5".format(r) in f for r in runs):
                    run = int(f.split("run")[-1].split(".h5")[0])
                    run_dict[run]["t2_path"] = "{}/{}".format(p,f)

        # check if files already exist
        pprint(run_dict)




        # get pygama build options for each run

        # for conf in runDB["build_options"]:
            # print(runDB["build_options"][conf]["coverage"])

        # if verbose:
        #     print("Tier 0:",len(t0_files),"files:")
        #     for f in t0_files:
        #         print(f)
        #
        #     print("Tier 1:",len(t1_files),"files:")
        #     for f in t1_files:
        #         print(f)
        #
        #     print("Tier 2:",len(t2_files),"files:")
        #     for f in t0_files:
        #         print(f)