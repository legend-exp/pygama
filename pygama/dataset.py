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
            self.raw_dir = os.path.expandvars(self.runDB["raw_dir"])
            self.tier_dir = os.path.expandvars(self.runDB["tier_dir"])
            self.t1pre = self.runDB["t1_prefix"]
            self.t2pre = self.runDB["t2_prefix"]
        except:
            print("Bad metadata, reverting to defaults ...")
            self.raw_dir = raw_dir
            self.tier_dir = tier_dir
            self.t1pre = "t1_run"
            self.t2pre = "t2_run"

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
        self.get_paths(self.runs, v)

        # could store concatenated dfs here, like a TChain
        self.df = None

    def load_metadata(self, fname):
        """
        load a JSON file into a dict
        """
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
        self.paths = {r:{} for r in runs}

        # search data directories for extant files
        for p, d, files in os.walk(self.raw_dir):
            for f in files:
                if any("Run{}".format(r) in f for r in runs):
                    run = int(f.split("Run")[-1])
                    self.paths[run]["t0_path"] = "{}/{}".format(p,f)

        for p, d, files in os.walk(self.tier_dir):
            for f in files:
                if any("t1_run{}".format(r) in f for r in runs):
                    run = int(f.split("run")[-1].split(".h5")[0])
                    self.paths[run]["t1_path"] = "{}/{}".format(p,f)

                if any("t2_run{}".format(r) in f for r in runs):
                    run = int(f.split("run")[-1].split(".h5")[0])
                    self.paths[run]["t2_path"] = "{}/{}".format(p,f)

        # get pygama build options for each run
        if self.runDB is not None:
            cov = {}
            for conf in self.runDB["build_options"]:
                cov[conf] = self.runDB["build_options"][conf]["coverage"]

            for run in runs:
                for conf, ranges in cov.items():
                    if ranges[0] <= run <= ranges[1]:
                        self.paths[run]["build_opt"] = conf

        # check for missing entries
        # TODO: get lists of unprocessed runs?
        for r in runs:
            if "t0_path" not in self.paths[r].keys():
                self.paths[r]["t0_path"] = None
            if "t1_path" not in self.paths[r].keys():
                self.paths[r]["t1_path"] = None
            if "t2_path" not in self.paths[r].keys():
                self.paths[r]["t2_path"] = None
            if "build_opt" not in self.paths[r].keys():
                self.paths[r]["build_opt"] = None

    def get_t1df(self):
        """
        concat tier 1 df's.
        careful, it can be a lot
        to load in memory ...
        """
        dfs = []
        for run in self.runs:
            p = self.paths[run]["t1_path"]
            dfs.append(pd.read_hdf(p))
        return pd.concat(dfs)

    def get_t2df(self):
        """
        concat tier 2 dfs.
        """
        dfs = []
        for run in self.runs:
            p = self.paths[run]["t2_path"]
            dfs.append(pd.read_hdf(p))
        return pd.concat(dfs)

    def get_runtime(self):
        """
        get the runtime (in seconds)
        of all runs in the current DataSet.
        """
        # pprint(self.paths)
        rt = 0
        for run in self.runs:
            p = self.paths[run]["t2_path"]
            df = pd.read_hdf(p)
            print(df.columns)
            print(df["ts_hi"][0]) # nope
            exit()
