import os
import json
import pandas as pd
import string
import numpy as np
import h5py
from parse import parse
from pygama import WaveformBrowser


class DataLoader:
    """
    Class to facilitate analysis of pygama-processed data across several tiers,
    daq->raw->dsp->hit->evt.  Where possible, we use a SQL-style database of
    cycle files so that a user can quickly select a subset of cycle files for
    interest, and access information at each processing tier.
    Includes methods to build a cycleDB, scan available parameter names in each
    file, and available tables (detectors).
    """
    def __init__(self, config=None, cycleDB=None, cycleDB_config=None, cycle_query:str=None):
        """
        DataLoader init function.  No hit-level data is loaded in memory at
        this point.  User should specify a config file containing DAQ filename
        format specifiers, etc.

        Parameters
        ----------
        config : dict or filename of JSON input file
            add description here
        cycleDB : pd.DataFrame, or filename of existing cycleDB
            add description here
        cycle_query : str
            String query that should operate on columns of a cycleDB.

        Returns
        -------
        None.
        """
        # declare all member variables
        self.config = None          # dict
        self.cycleDB = None         # pygama CycleDB
        self.cycle_list = None      # pygama CycleDB
        self.table_list = None      # array-like of strings
        self.cuts = np.empty(0)     # array-like of strings
        self.entry_list = None      
        self.merge_cycles = False
        self.output_format = 'lgdo.Table'
        self.output_columns = None 

        # load things if available
        if config is not None:
            if isinstance(config, str):
                with open(config) as f:
                    config = json.load(f)
            self.set_config(config) 
            

        if cycleDB is None:
            if cycleDB_config is None:
                print("Either cycleDB or cycleDB_config is required!")
                return
            else:
                self.cycleDB = CycleDB(cycleDB_config)
        else:
            if isinstance(cycleDB, pd.DataFrame):
                self.cycleDB = CycleDB(cycleDB)
            elif isinstance(cycleDB, str):
                self.cycleDB = pd.read_hdf(cycleDB, key='cycleDB')
            else:
                print("cycleDB must be a string or instance of cycleDB")
                

        if cycle_query is not None:
            # modify the given cycleDB (maybe in-place or not)
            self.cycleDB = self.cycleDB.query(cycle_query, inplace=True)


    def set_config(self, config:dict):
        """
        load JSON config file
        """
        self.config = config 
        self.data_dir = config['data_dir']
        self.tiers = config['tiers']
        
        self.dt = config['daq_template']
        self.di = config['daq_ignore']
        self.daq_dir = config['daq_dir']


    def set_cycles(self, query:str):
        """
        Set the files of interest, do this before any other operations

        Parameters
        ----------
        query : string 
            The cycle level cuts on the files of interest
            Can be a cut on any of the columns in CycleDB

        Returns
        -------
        None.
        """
        self.cycle_list = self.cycleDB.query(query) 

    def get_col_names(self, query:str, f_output:str=None) -> dict:
        """
        Return dict of cols {parameter : tier}
        Optionally write it to JSON or other output file
        """
        cycle = self.cycle_list.iloc[0]
        with cycle['raw_file'] open as f:
            x = f.keys()

            
        return {}

    def set_datastreams(self, ds=None):
        """
        Set the datastreams (detectors) of interest

        Parameters
        -----------
            ds: array-like of strings
            detector_ids or channels of interest

        """
        self.table_list = {x: {tier: '' for tier in self.tiers} for x in ds}

    def get_table_names():
        """
        Save to dictionary or similar
        """
        self.cycle_list 
        pass


    def set_cuts(self, cut=None):
        """
        Set the hit- or event-level cuts

        Parameters
        ----------
        cut : array-like of strings
            The cuts on the columns of the data table, e.g. "trapEftp_cal > 1000"
            Each item in the array should be able to be applied on one tier of tables, 
            as specified in config['joinable'] 

        Returns
        -------
        None.
        """


    def set_cycle_status():
        """
        using a file list, scan for the existence of [daq, raw, dsp, hit, evt]
        cycle files, and pack them into an integer.  Augment self.cycleDB with
        this result in a column.
        Example: [0   1   1   0   0]
                 daq raw dsp hit evt
        """
        pass


    def set_output_format():
        """
        lgdo.Table, pd.DataFrame, awkward-array, ROOT, ... others?
        """
        pass


    def skim_hits():
        """
        use self.cycle_list to get filenames,
        and self.table_list to get table names.
        Return a memory structure matching `set_output_format`
        Should be able to either: load into memory, or write to an output file.
        Important - do a chunked read & write, don't try to read everything
        into memory all at once.
        """
        pass


    def skim_events():
        """
        same comment as skim_hits, except we return an event-list formatted table
        """
        pass


    def skim_waveforms(mode:str='hit', hit_list=None, evt_list=None):
        """
        handle this one separately because waveforms can easily fill up memory.
        """
        if mode='hit':
            pass
        elif mode='evt':
            pass
        pass


    def browse(query, dsp_config=None):
        """
        Interface between DataLoader and WaveformBrowser.
        """
        wb = WaveformBrowser()
        return wb

    def gen_entry_list():
        """
        This
        """


    def load(query, cuts, columns):
        """
        this should load everything we need from cycleDB, table names, columns,
        etc, but should probably NOT load data into memory - that's the job
        of skim_hits and skim_events
        """
        pass


    def load_detector(det_id):
        """
        special version of `load` designed to retrieve all cycle files, tables,
        column names, and potentially calibration/dsp parameters relevant to one
        single detector.
        """
        pass


    def load_settings():
        """
        get metadata stored in raw files, usually from a DAQ machine.
        """
        pass


    def load_dsp_pars(query):
        """
        access the dsp_pars parameter database (probably JSON format) and do
        some kind of query to retrieve parameters of interest for our cycle list,
        and return some tables.
        """
        pass


    def load_cal_pars(query):
        """
        access the cal_pars parameter database, run a query, and return some tables.
        """
        pass


class CycleDB(pd.DataFrame):
    """
    A pandas DataFrame that has additional functions to scan the data directory,
    fill its own columns with information about each cycle, and
    read/write to disk in an LGDO format
    """
    def __init__(self, config=None, file=None, scan=True):
        """
        Parameters
        ----------
            config : path to JSON file or dict
            Configuration file specifying data directories and tiers

            file : string 
            Path to a file containing a LGDO.Table written out by CycleDB.to_lgdo()

            scan : bool
            True by default, whether the cycleDB should scan the DAQ directory to
            fill its rows with cycle information
        """

        if file is None:
            if config is None:
                print('Need to specify a configuration file or a file containing a cycleDB!')
                return

            if isinstance(config, str):
                with open(config) as f:
                    config = json.load(f)                           
        
            self.config = config
            self.tiers = list(config["tier_dirs"].keys())
            self.file_key = config["file_key"]
            self.daq_template = config["daq_template"]
            self.daq_dir = config["daq_dir"]
            self.data_dir = config["data_dir"]
            self.tier_dirs = config["tier_dirs"]
            # Set up column names
            names = list(parse(self.file_key, self.file_key).named.keys()) # fields required to generate file name
            names += [f'{tier}_file' for tier in self.tiers] # the generated file names
            names += [f'{tier}_size' for tier in self.tiers] # file sizes
            names += ['cycle_status', 'geds', 'calibration'] # bonus columns 

            super.__init__(columns=names)

            if scan:
                self.scan_cycles()
        else:
            self.from_lgdo(file)

        

    def scan_cycles(self, verbose=False):
        """
        Scan the DAQ directory and fill the DataFrame
        Only fills columns that can be populated with just the DAQ file
        """
        daq_dir = self.daq_dir 

        file_keys = []
        n_files = 0

        for path, folders, files in os.walk(daq_dir):
            n_files += len(files)

            for f in files:

                # in some cases, we need information from the path name
                if '/' in self.daq_template:
                    f_tmp = path.replace(self.daq_dir,'') + '/' + f
                else:
                    f_tmp = f

                finfo = parse(self.daq_template, f_tmp).named
                if finfo is not None:
                    finfo['daq_dir'] = path.replace(self.daq_dir,'') # sub-dir
                    finfo['daq_file'] = f
                    file_keys.append(finfo)

                for tier in self.tiers:
                    if tier == "daq":
                        continue

                    finfo[f'{tier}_file'] = self.file_key.format(**finfo)


        if n_files == 0:
            print("no daq files found...")
            return

        if len(file_keys) == 0:
            print("no daq files matched pattern", self.daq_template)
            return

        # fill the main DataFrame
        self.append(file_keys)

        # convert cols to numeric dtypes where possible
        for col in self.cycleDB.columns:
             try:
                self[col] = pd.to_numeric(self[col])
             except:
                pass
        
        if verbose:
            print(self)


    def show(self, col_names:list=None):
        """
        show the existing cycleDB as a DataFrame, optionally specifying columns
        """
        if col_names is None:
            print(self.cycleDB)
        else:
            print(self.cycleDB[col_names])

    def from_lgdo(self, fname):
        """
        Fills this DataFrame with the information from a file created by to_lgdo()
        """
        pass

    def to_lgdo(self, fname, update=True):
        """
        Converts this DataFrame to an lgdo.Table, and writes it to disk 
        along with the config information
        """

        pass

if __name__=='__main__':
    doc="""
    Demonstrate usage of the `DataLoader` class.
    This could be what we initially run at LNGS - it would try to do the `os.walk`
    method over the existing files, and e.g. scan for existence of various files
    in different stages.  More advanced tests would be moved to a notebook or
    separate script.
    """
    f_config = 'loader_config.json'

    dl = DataLoader(f_config)

    dl.scan_cycles()
    dl.set_cycle_status()
    dl.get_col_names()
    dl.get_table_names()

    dl.show_cycleDB()
    dl.save_cycleDB()

    dl.set_cycles(query='date == 2022-06-03 and type=="cal"')
    dl.set_datastreams(ds=['g024'])
    dl.set_cuts(cut='daqenergy > 100')
    el = dl.gen_entry_list()
    el.saveto('file.lh5')
    df = dl.load(entry_list=el, merge_cycles=True, fmt=pd.DataFrame, in_mem=True, columns=['trapEftp', 'AoE'])