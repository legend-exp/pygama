#!/usr/bin/env python3
import os, json
import numpy as np
import pandas as pd
import tinydb as db
from pprint import pprint

class DataSet:
    """
    this is based on GATDataSet.
    can initialize with data sets, a single run, or lists of runs.
    can also load a JSON file to get a dict of metadata.
    """
    def __init__(self, ds_lo=None, ds_hi=None, run=None, runlist=None, sub=None,
                 opt=None, v=False, md=None, cal=None, raw_dir=None, tier1_dir=None, tier2_dir=None ):

        # load metadata and set paths to data folders
        self.config, self.calDB = None, None
        if md is not None:
            self.load_metadata(md) # pure JSON
        if cal is not None:
            self.calDB = db.TinyDB(cal) # TinyDB JSON
        try:
            self.raw_dir = os.path.expandvars(self.config["raw_dir"])
            self.t1pre = self.config["t1_prefix"]
            self.t2pre = self.config["t2_prefix"]

        except:
            print("Bad metadata, reverting to defaults ...")
            self.raw_dir = raw_dir
            self.t1pre = "t1_run"
            self.t2pre = "t2_run"

        self.tier_dir = None
        self.tier1_dir = os.path.expandvars(self.config["tier1_dir"])
        self.tier2_dir = os.path.expandvars(self.config["tier2_dir"])

        try:
            self.ftype = self.config["filetype"]
        except:
            self.ftype = "default"

        # match ds number to run numbers
        self.ds_run_table = {}
        for ds in self.config["ds"]:
            try:
                dsnum = int(ds)
            except:
                continue
            run_cov = self.config["ds"][ds][0].split(",")
            self.ds_run_table[int(ds)] = [int(r) for r in run_cov]

        # create the internal lists of run numbers and ds's
        self.runs, self.ds_list = [], []
        if ds_lo is not None:
            self.runs.extend(self.get_runs(ds_lo, ds_hi, v))
            self.ds_lo = ds_lo
        if run is not None:
            self.runs.append(run)
            self.ds_list.append(self.lookup_ds(run))
        if runlist is not None:
            self.runs.extend(runlist)
            self.ds_list.extend([self.lookup_ds(r) for r in runlist])
        if opt == "-all":
            self.runs.extend(self.get_runs(verbose=v))

        if v:
            print("Found run numbers:", self.runs)

        # filenames for every run
        self.get_paths(self.runs,sub, v)

        # could store concatenated dfs here, like a TChain
        self.df = None


    def load_metadata(self, fname):
        """
        load a JSON file into a dict
        """
        with open(fname) as f:
            self.config = json.load(f)


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
        using the config,
        create a list of data sets to process,
        then return a list of the included run numbers
        """
        if self.config is None:
            print("Error, config not set.")
            return []

        # load all data
        if ds_lo is None and ds_hi is None:
            self.ds_list.extend([d for d in self.config["ds"] if d != "note"])

        # load single ds
        elif ds_hi is None:
            self.ds_list.append(ds_lo)

        # load ds range
        else:
            self.ds_list.extend([str(d) for d in range(ds_lo, ds_hi+1)])

        run_list = []
        for ds in self.ds_list:
            tmp = self.config["ds"][str(ds)][0].split(",")
            r1 = int(tmp[0])
            r2 = int(tmp[1]) if len(tmp)>1 else None
            if r2 is None:
                run_list.append(r1)
            else:
                run_list.extend([r for r in range(r1, r2+1)]) # inclusive

        if verbose:
            print("Data Sets:",self.ds_list)
            print("Runs:",run_list)

        return run_list


    def get_paths(self, runs, subfile, verbose=False):
        """
        TODO: REFACTOR THIS ROUTINE TO BE SHORTER AND MORE GENERAL.

        Matteo -- add "get_file_list", it will work with all sets of input files

        collect path info and flag nonexistent files.
        does a directory search with os.walk, which is faster than iglob
        https://stackoverflow.com/questions/1724693/find-a-file-in-python

        Note from Andreas --
        A new filetype flag in the runDB:
        In the HADES characterization campaign we have a different filestructure and we have Subfiles.
        e.g. our HADES HV-Scan data consists of 5 Files per Run and the file names are structured like:
        "char_data-DetID-Source-run00XY-YYMMDDTHHMMSS.fcio"
        Also the are ".log" files in the same directory
        To deal with different filenames I added a filteype flag in the runDB. If it is hades_char I look for HADES structure.
        Other structure are not implemented. If no filetype flag is given, the original files earch is done.
        What I still don't like is that I use counter for the subfiles. I have to think about something better.
        """
        self.paths = {r:{} for r in runs}

        # choose method of searching for raw files -- depends on file fmt string
        if self.ftype == "hades_char":
           # search data directories for extant files

           # Check for raw Data
           counter = 1
           for p, d, files in os.walk(self.raw_dir):
               for f in files:
                   for r in runs:
                       if f.endswith(".fcio"):
                           if int(f.split("run")[-1].split("-")[0]) == r:
                               if counter==subfile:
                                  run = int(f.split("run")[-1].split("-")[0])
                                  self.paths[run]["t0_path"] = "{}/{}".format(p,f)
                               counter+=1

           # Check for tier1 Data
           counter = 1
           for p, d, files in os.walk(self.tier1_dir):
               for f in files:
                   if any("{}-".format(r) in f for r in runs):
                       if counter == subfile:
                          run = int(f.split("run")[-1].split("-")[0])
                          self.paths[run]["t1_path"] = "{}/{}".format(p,f)
                       counter += 1


           # Check for tier2 Data
           counter = 1
           for p, d, files in os.walk(self.tier2_dir):
               for f in files:
                   if any("{}-".format(r) in f for r in runs):
                       if counter == subfile:
                          run = int(f.split("run")[-1].split("-")[0])
                          self.paths[run]["t2_path"] = "{}/{}".format(p,f)
                       counter += 1




        elif self.ftype == "legend200":
            print("Read awsome LEGEND200 Data. But not ready yet...")

        elif self.ftype == "flashcam":

            # .lh5
            suffix = "." + self.config["suffix"]

            for p, d, files in os.walk(self.raw_dir):
                for f in files:
                    if any("protothppmco_{}".format(r) in f for r in runs):
                        run = f.split("protothppmco_")[-1]
                        run = run.split(".")[0]
                        if ".log" in run:
                            continue
                        else:
                            run = int(run)
                        self.paths[run]["t0_path"] = "{}/{}".format(p,f)

            for p, d, files in os.walk(self.tier1_dir):
                for f in files:
                    if any("t1_run{}".format(r) in f for r in runs):
                        run = int(f.split("run")[-1].split(suffix)[0])
                        self.paths[run]["t1_path"] = "{}/{}".format(p,f)

            for p, d, files in os.walk(self.tier2_dir):
                for f in files:
                    if any("t2_run{}".format(r) in f for r in runs):
                        run = int(f.split("run")[-1].split(suffix)[0])
                        self.paths[run]["t2_path"] = "{}/{}".format(p,f)

        else:
	
            suffix = "." + self.config["suffix"]#edit: we need this also here. One has to set the suffix then
                        # in the config file ("suffix":"h5")

            # search data directories for extant files
            for p, d, files in os.walk(self.raw_dir):
                for f in files:
                    if any("Run{}".format(r) in f for r in runs):
                        run = f.split("Run")[-1]
                        if ".log" in run:
                            continue
                        else:
                            run = int(run)
                        self.paths[run]["t0_path"] = "{}/{}".format(p,f)

            for p, d, files in os.walk(self.tier1_dir):
                for f in files:
                    if any("t1_run{}".format(r) in f for r in runs):
                        run = int(f.split("run")[-1].split(suffix)[0])
                        self.paths[run]["t1_path"] = "{}/{}".format(p,f)

            for p, d, files in os.walk(self.tier2_dir):
                for f in files:
                    if any("t2_run{}".format(r) in f for r in runs):
                        run = int(f.split("run")[-1].split(suffix)[0])
                        self.paths[run]["t2_path"] = "{}/{}".format(p,f)

        # get pygama build options for each run
        if self.config is not None:
            cov = {}
            for conf in self.config["build_options"]:
                cov[conf] = self.config["build_options"][conf]["run_coverage"]

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


    def lookup_ds(self, run):
        """
        given a run number, figure out what data set it belongs to.
        """
        for ds in self.ds_run_table:
            runlist = self.ds_run_table[ds]
            if len(runlist) == 1 and run == runlist[0]:
                return ds
            elif len(runlist) > 1 and runlist[0] <= run <= runlist[-1]:
                return ds

        # if we get to here, we haven't found the run
        print("Error, couldn't find a ds for run {run}.")
        exit()


    def get_p1cal_pars(self, etype):
        """
        return the pass-1 initial guess parameters for an energy estimator.
        """
        for key in self.config["ecal"]:
            tmp = key.split(",")
            if len(tmp) == 1:
                continue
            ds_lo, ds_hi = int(tmp[0]), int(tmp[1])

            tmp2 = np.array(self.ds_list)
            iout = np.where((tmp2 < ds_lo) | (tmp2 > ds_hi))
            if len(iout[0]) > 0:
                print("Error, we don't currently support multiple p1 cal pars.")
                exit()
            # print(key,etype)
            pars = self.config["ecal"][key][etype]
            # pprint(pars)
            return pars


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


    def get_ts(self, df=None, clock=1e8, rollover=False, test=False):
        """
        return an ndarray of timestamps (in seconds) in the run.
        - handle timestamps which roll over (SIS3302) and ones that don't.
        - to avoid iterating over every element, we identify contiguous
          blocks by looking for when the previous TS value is greater than the
          current one, and then add the appropriate increment block by block.
        """
        if df is None:
            # be careful, this can load multiple runs
            df = self.get_t2df()

        if not rollover:
            return df["timestamp"].values / clock # seconds

        # Maximum value for a variable of type unsigned int.
        UINT_MAX = 4294967295 # (0xffffffff)
        t_max = UINT_MAX / clock

        ts = df["timestamp"].values / clock
        tdiff = np.diff(ts)
        tdiff = np.insert(tdiff, 0 , 0)
        entry = np.arange(0, len(ts), 1)
        iwrap = np.where(tdiff < 0)
        iloop = np.append(iwrap[0], len(ts))

        ts_new, t_roll = [], 0

        for i, idx in enumerate(iloop):

            ilo = 0 if i==0 else iwrap[0][i-1]
            ihi = idx

            ts_block = ts[ilo:ihi]
            t_last = ts[ilo-1]
            t_diff = t_max - t_last
            ts_new.append(ts_block + t_roll)

            t_roll += t_last + t_diff # increment for the next block

        ts_wrapped = np.concatenate(ts_new)

        if test:
            # make sure timestamps are continuously increasing vs entry number
            import matplotlib.pyplot as plt
            plt.plot(ts_wrapped, entry, "-b")
            plt.xlabel("Time (s)", ha='right', x=1)
            plt.ylabel("Entry Num", ha='right', y=1)
            plt.tight_layout()
            plt.show()
            # exit()

        return ts_wrapped


    def get_runtime(self, clock=None, rollover=None):
        """
        get the runtime (in seconds)
        of all runs in the current DataSet.
        NOTE: right now i get it by taking the difference
        of the last and first timestamp.
        This is wrong by a factor ~2*tau (dt between events).
        """
        if clock is None:
            clock = self.config["clock"]
        if rollover is None:
            rollover = self.config["rollover"]

        total_rt = 0
        for run in self.runs:
            p = self.paths[run]["t2_path"]
            df = pd.read_hdf(p)
            ts = self.get_ts(df, clock, rollover)

            # here's where we could put in extra factors such as 2*tau
            rt = ts[-1] - ts[0]

            total_rt += rt

        return total_rt


    def daq_to_raw(self, overwrite=False, test=False, n_max=np.inf):
        """
        convenience function for calling the main daq_to_raw function.
        right now, this processes runs sequentially.
        """
        from pygama.io.daq_to_raw import daq_to_raw

        for run in self.runs:
            t0_file = self.paths[run]["t0_path"]
            t1_file = self.paths[run]["t1_path"]
            if t1_file is not None and overwrite is False:
                print("file exists, overwrite flag isn't set.  continuing ...")
                continue
            if test:
                print("test mode (dry run), processing Tier 0 file:\n    ", t0_file)
                continue

            daq_to_raw(t0_file, run, suffix=self.config["suffix"],
                       verbose=test, output_dir=self.tier1_dir,
                       overwrite=overwrite, n_max=n_max, config=self.config)


    def run_dsp(self, overwrite=False, test=False, proc_list=None, out_dir=None,
                verbose=False, multiproc=True):
        """
        output a file with dsp parameters
        """
        from pygama.dsp.base import Intercom
        from pygama.io.tier1 import ProcessTier1

        for run in self.runs:
            t1_file = self.paths[run]["t1_path"]
            t2_file = self.paths[run]["t2_path"]
            if t2_file is not None and overwrite is False:
                continue

            if test:
                print("test mode (dry run), processing Tier 1 file:", t1_file)
                continue

            conf = self.paths[run]["build_opt"]

            if proc_list is None:
                proc_list = self.config['build_options'][conf]['tier1_options']

            proc = Intercom(proc_list)

            out_dir = self.tier2_dir if out_dir is None else out_dir

            ProcessTier1(t1_file, proc, output_dir=out_dir,
                         overwrite=overwrite, verbose=verbose,
                         multiprocess=False, nevt=np.inf,
                         ioff=0, chunk=self.config["chunksize"])
