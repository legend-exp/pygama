import os
import json
import pandas as pd
import string
import re
import numpy as np
import h5py
import time
from parse import parse
from pygama.lgdo import *
#from pygama import WaveformBrowser


class DataLoader:
    """
    Class to facilitate analysis of pygama-processed data across several tiers,
    daq->raw->dsp->hit->evt.  Where possible, we use a SQL-style database of
    cycle files so that a user can quickly select a subset of cycle files for
    interest, and access information at each processing tier.
    Includes methods to build a fileDB, scan available parameter names in each
    file, and available tables (detectors).
    """
    def __init__(self, config=None, fileDB=None, fileDB_config=None, file_query:str=None):
        """
        DataLoader init function.  No hit-level data is loaded in memory at
        this point.  User should specify a config file containing DAQ filename
        format specifiers, etc.

        Parameters
        ----------
        config : dict or filename of JSON config file
            add description here
        fileDB : pd.DataFrame, FileDB, or filename of existing fileDB
            A fileDB must be specified, either with
                1) An instance of FileDB
                2) files written by FileDB.to_disk() (both fileDB and fileDB_config)
                3) config file with enough info for FileDB to perform a DAQ scan
                4) pd.DataFrame with a config file
        fileDB_config : dict or filename of JSON config file
            Config file mentioned above for fileDB
        file_query : str
            String query that should operate on columns of a fileDB.

        Returns
        -------
        None.
        """
        # declare all member variables
        self.config = None          # dict
        self.fileDB = None         # pygama FileDB
        self.file_list = None      # list
        self.table_list = None      
        self.cuts = None     
        self.merge_files = False
        self.output_format = 'lgdo.Table'
        self.output_columns = None 

        # load things if available
        if config is not None:
            if isinstance(config, str):
                with open(config) as f:
                    config = json.load(f)
            self.set_config(config) 

        if fileDB is None:
            if fileDB_config is None:
                print("Either fileDB or fileDB_config is required!")
                return
            else:
                self.fileDB = FileDB(config=fileDB_config)
        else:
            if isinstance(fileDB, pd.DataFrame) or isinstance(fileDB, str):
                if fileDB_config is None:
                    print("Must provide a config file with a fileDB dataframe")
                    return
                self.fileDB = FileDB(config=fileDB_config, file_df=fileDB)
            elif isinstance(fileDB, FileDB):
                self.fileDB = fileDB
            else:
                print("fileDB must be a string or instance of fileDB or pd.DataFrame")
        
        if file_query is not None:
            # set the file_list
            self.file_list = list(self.fileDB.df.query(file_query).index)

    def set_config(self, config:dict):
        """
        load JSON config file
        """
        self.config = config 
        self.data_dir = config["data_dir"]
        self.levels = list(config["levels"].keys())
        self.tiers = {}
        self.cut_priority = {}
        self.tcm_cols = {}
        for level in self.levels:
            self.tiers[level] = config["levels"][level]["tiers"]
            #Set cut priority
            if "dependency" in config["levels"][level].keys():
                dep = config["levels"][level]["dependency"]
                self.cut_priority[level] = self.cut_priority[dep] + 1
                #Set TCM columns to lookup
                if "tcm_cols" in config["levels"][level].keys():
                    self.tcm_cols[level] = config["levels"][level]["tcm_cols"]
                else:
                    print(f"Config Warning: Levels dependent on lower levels, e.g. {level}, need to specify the TCM lookup columns")
            else:
                self.cut_priority[level] = 0
                
        
        #Set channel map
        if isinstance(config["channel_map"], dict):
            self.channel_map = config["channel_map"]
        elif isinstance(config["channel_map"], str):
            with open(config["channel_map"]) as f:
                self.channel_map = json.load(f)
        else:
            print("Config Warning: Channel map must be dict or path to JSON file")

    def set_files(self, query:str):
        """
        Set the files of interest, do this before any other operations
        self.file_list is a list of indices corresponding to the row in FileDB

        Parameters
        ----------
        query : string 
            The file level cuts on the files of interest
            Can be a cut on any of the columns in FileDB

        Returns
        -------
        None.
        """
        inds = list(self.fileDB.df.query(query, inplace=False).index)    
        if self.file_list is None:
            self.file_list = inds
        else:
            self.file_list += inds

    def set_datastreams(self, ds, word): #TODO Make this able to handle more complicated requests
        """
        Set the datastreams (detectors) of interest

        Parameters
        -----------
            ds : array-like 
            Identifies the detectors of interest
            Can be a list of detectorID, serialno, or channels
            or a list of subsystems of interest e.g. "ged" 

            word : string
            The type of identifier used in ds 
            Should be a key in the given channel map or a word in the config file

        table_list = {
            "hit": [0, 1, 2]
            "evt": []        
        }
        
        As far as I know there is only one table per evt file. 
        We want to be able to handle things more generally, but for now let's just support setting "channel".
        """
        if self.table_list is None:
            self.table_list = {}

        found = False
        for level in self.levels:
            tier = self.tiers[level][0]
            table_template = self.fileDB.table_format[tier] 
            keyword = list(parse(table_template, table_template).named.keys())[0] 
            if word == keyword:
                found = True
                if level in self.table_list.keys():
                    self.table_list[level] += ds 
                else:
                    self.table_list[level] = ds

        if not found:
            #look for word in channel map
            pass

    def set_cuts(self, cuts):
        """
        Set the hit- or event-level cuts

        Parameters
        ----------
        cut : dictionary or list of strings
        The cuts on the columns of the data table, e.g. "trapEftp_cal > 1000"
        If passing a dictionary, the dictionary should be structured the way that cuts 
        will be stored in memory
        If passing a list, each item in the array should be able to be applied on one level of tables, 
        in the order specified in config['levels'],
        The cuts at different levels will be joined with an "and"

        e.g. if the full cut is "trapEmax > 1000 and lar_veto == False and dcr < 2" 
        list: ["lar_veto == False", "trapEmax > 1000 and dcr < 2"] (order matters)
        dictionary:
        cuts:{
            "hit": "trapEmax > 1000 and dcr < 2",
            "evt": "lar_veto == False"
        }

        Returns
        -------
        None.
        """
        if self.cuts is None:
            self.cuts = {}
        if isinstance(cuts, dict):
            # verify the correct structure
            for key, value in cuts.items():
                if not(key in self.levels and isinstance(value, str)):
                    print("Error: cuts dictionary must be in the format \{ level: string \}")
                    return 
                if key in self.cuts.keys():
                    self.cuts[key] += " and " + value 
                else:
                    self.cuts[key] = value
        elif isinstance(cuts, list):
            self.cuts = {}
            # TODO Parse strings to match column names so you don't have to specify which level it is

    def set_output(self, fmt=None, merge_files=None, columns=None):
        """
        Parameters
        ----------
        fmt : string
        'lgdo.Table', 'pd.DataFrame', or TBD
        Defaults to lgdo.Table

        merge_files : bool
        If true, information from multiple files will be merged into one table

        columns : array-like of strings
        The columns that should be copied into the output

        Returns
        -------
        None.
        """
        if fmt is not None:
            self.output_format = fmt 
        if merge_files is not None:
            self.merge_files = merge_files 
        if columns is not None:
            self.output_columns = columns 

    def show_file_list(self, columns=None):
        if columns is None:
            print(self.fileDB.df.iloc[self.file_list])
        else:
            print(self.fileDB.df[columns].iloc[self.file_list])

    def show_fileDB(self, columns=None):
        self.fileDB.show(columns)   

    def gen_entry_list(self, chunk=False, mode='only'): #TODO: mode, chunking, etc
        """
        This should apply cuts to the tables and files of interest
        but it does NOT load the column information into memory
        
        Parameters
        ----------
        chunk : bool ?????????????????
        If true, iterates through each file in file_list
        If false, opens all files at once 

        mode : 'any' or 'only'
        'any' : returns every hit in the event if any hit in the event passes the cuts
        'only' : only returns hits that pass the cuts

        Returns
        -------
        entries:  
        -------------------------------
        event   |   channel |   row 
        -------------------------------
        0           5           0
        0           6           0
        0           12          0
        1           5           1
        2           5           2
        2           6           1

        """
        if self.file_list is None:
            print("You need to make a query on fileDB, use set_file_list")
            return 
        
        entries = {}

        # Default columns in the entry_list
        entry_cols = [f"{level}_table" for level in self.levels]
        entry_cols += [f"{level}_idx" for level in self.levels]

        # Find out which columns are needed for the cut
        cut_cols = {}
        for level in self.levels:
            cut_cols[level] = []
            if self.cuts is not None and level in self.cuts.keys():
                cut = self.cuts[level] 
            else: 
                cut = ""
            # String parsing to determine which columns need to be loaded
            split = re.split(' |<|>|=|and|or|&|\|', cut) 
            for term in split:
                if term.isidentifier(): #Assumes that column names are valid python variable names
                    cut_cols[level].append(term)
                    # Add column to entry_cols if they are needed for both the cut and the output
                    if self.output_columns is not None:
                        if term in self.output_columns and term not in for_output:
                            entry_cols.append(term)
        
        # Make the entry list for each file
        for file in self.file_list:
            f_entries = pd.DataFrame(columns=entry_cols)
            # Get levels needed for cuts, and sort by cut priority
            cut_levels = sorted(list(self.cuts.keys()), key=lambda level: self.cut_priority[level], reverse=True)
            first_cut_made = False
            for level in cut_levels:
                tables = []
                level_paths = {}
                for tier in self.tiers[level]:
                    path = self.data_dir + self.fileDB.tier_dirs[tier] + '/' + self.fileDB.df.iloc[file][f'{tier}_file']
                    # Only add to level_paths if the file exists
                    if os.path.exists(path):
                        level_paths[tier] = path
                        if not tables:
                            # Tables should be shared across tiers in one level
                            tables = self.fileDB.df.iloc[file][f'{tier}_tables']
                
                if self.table_list is not None:
                    if level in self.table_list.keys():
                        tables = self.table_list[level]

                # Continue if the paths exist
                if level_paths:
                    cut = self.cuts[level]
                    level_table = None 
                    fields = []
                    # Use tcm_cols to find column names that correspond to level_table and level_idx 
                    if level in self.tcm_cols.keys():
                        fields = list(self.tcm_cols[level].values())

                    sto = LH5Store()
                    for tb in tables:
                        # Try to create index mask for this table
                        try:
                            idx = f_entries.query(f"{level}_table == {tb}")[f"{level}_idx"]
                        except:
                            idx = None

                        for tier, path in level_paths.items():
                            template = self.fileDB.table_format[tier]
                            keyword = list(parse(template, template).named.keys())[0]
                            args = {keyword: tb}
                            table_name = template.format(**args)
                            temp_table, _ = sto.read_object(table_name, path, idx=idx, field_mask=fields + cut_cols)
                            if level_table is None:
                                level_table = temp_table
                            else:
                                level_table.join(temp_table) 
                        level_df = level_table.get_dataframe()
                        cut_df = level_df.query(cut)
                        # rename columns to match entry_cols
                        if level in self.tcm_cols.keys():
                            dep = self.config["levels"][level]["dependency"]
                            renaming = {
                                self.tcm_cols[level]["table_col"]: f"{dep}_table", 
                                self.tcm_cols[level]["idx_col"]: f"{dep}_idx", 
                            }
                            cut_df = cut_df.rename(renaming, axis="columns")
                            cut_df = cut_df.explode(list(renaming.values()), ignore_index=True)
                        else:
                            cut_df[f"{level}_idx"] = cut_df.index
                            cut_df[f"{level}_table"] = tb
                        # THIS NEXT PART NEEDS WORK
                        if not first_cut_made:
                            f_entries = pd.concat((f_entries, cut_df), ignore_index=True)[entry_cols]
                        else:
                            tb_entries = f_entries.query(f"{level}_table == {tb}")
                            drop_entries = tb_entries.query(f"{level}_idx not in {list(cut_df.index)}")
                            f_entries = f_entries.drop(list(drop_entries.index))
                first_cut_made = True
####################################################################################################################################################### 
################################################################## Old Code Under HERE ################################################################
#######################################################################################################################################################
            # Grab file paths from fileDB
            for level in self.levels:
                level_paths = {}
                tables = []
                for tier in self.tiers[level]:
                    path = self.data_dir + self.fileDB.tier_dirs[tier] + '/' + self.fileDB.df.iloc[file][f'{tier}_file']
                    # Only add to level_paths if the file exists
                    if os.path.exists(path):
                        level_paths[tier] = path
                        if not tables:
                            tables = self.fileDB.df.iloc[file][f'{tier}_tables']

                if self.table_list is not None:
                    tables = self.table_list[level]

                if level_paths:
                    if self.cuts is not None and level in self.cuts.keys():
                        cut = self.cuts[level] 
                    else: 
                        cut = ""
                    

                    sto = LH5Store()
                    for tb in tables:
                        idx = None
                        try:
                            lev_prev = self.levels[i-1]
                            idx = entries[file][lev_prev][tb]
                        except:
                            pass 

                        level_table = None
                        for tier, path in level_paths.items():
                            template = self.fileDB.table_format[tier]
                            keyword = list(parse(template, template).named.keys())[0]
                            args = {keyword: tb}
                            tb_name = template.format(**args)
                            print(tb_name)
                            temp_tb, _ = sto.read_object(tb_name, path, idx=idx, field_mask=cut_cols)
                            if level_table is None:
                                level_table = temp_tb 
                            else: 
                                level_table.join(temp_tb)

                        level_df = level_table.get_dataframe() 
                        if cut:
                            cut_df = level_df.query(cut)
                        else:
                            cut_df = level_df 
                        if level in self.tcm_cols.keys(): 
                            entries[file][level][tb] = cut_df[self.tcm_cols[level]].to_list()
                        else:
                            entries[file][level][tb] = cut_df.index.to_list()


        return entries

    def load(self, entry_list=None, in_mem=False, f_output=None, rows='hit'): #TODO
        if rows == 'hit':
            return self.load_hits(entry_list, in_mem, f_output)
        elif rows == 'evt':
            return self.load_evts(entry_list, in_mem, f_output)
        else:
            print(f"I don't understand what rows={rows} means!")
            return

    def load_hits(self, entry_list=None, in_mem=False, f_output=None):
        """
        Actually retrieve the information from the events in entry_list, and 
        return it in the requested output format 
        """
        if entry_list is None:
            print("First run gen_entry_list and pass the output to load")
            return 

        if in_mem == False and f_output is None:
            print("If in_mem is False, need to specify an output file")
            return

        sto = LH5Store()
        writing = False
        if self.merge_files: # Try to load all information at once
            if in_mem:
                load_ret = Table()
            level = self.levels[0]
            tables = entry_list[self.file_list[0]][level].keys()
            for tb in tables:
                lowest_idx = [entry_list[file][level][tb] for file in self.file_list]
                tier_loads = []
                for tier in self.tiers[level]: 
                    template = self.fileDB.table_format[tier]
                    keyword = list(parse(template, template).named.keys())[0]
                    args = {keyword: tb}
                    tb_name = template.format(**args)
                    file_paths = []
                    for file in self.file_list:
                        path = self.data_dir + self.fileDB.tier_dirs[tier] + '/' + self.fileDB.df.iloc[file][f'{tier}_file']
                        if os.path.exists(path):
                            file_paths.append(path)
                    if file_paths:
                        tb_out, _ = sto.read_object(tb_name, file_paths, idx=lowest_idx, field_mask=self.output_columns)
                        tier_loads.append(tb_out)
                for i in range(len(tier_loads)-1):
                    tier_loads[0].join(tier_loads[i+1])
            
                if f_output:
                    if not writing:
                        writing = True
                        sto.write_object(tier_loads[0], f"load{tb}", f_output, wo_mode="overwrite_file")
                    else:
                        sto.write_object(tier_loads[0], f"load{tb}", f_output, wo_mode="append")
                if in_mem:
                    load_ret.add_column(f"load{tb}", tier_loads[0], use_obj_size=True)

            if in_mem:
                if self.output_format == "lgdo.Table":
                        return load_ret
                elif self.output_format == "pd.DataFrame":
                    return [tb.get_dataframe() for tb in load_ret.values()]
                else:
                    print("I don't know how to output " + self.output_format + ", here is a lgdo.Table")
                    return load_ret
            else:
                return
        else: #Not merge_files
            level = self.levels[0]
            load_ret = []
            for file, fdict in entry_list.items():
                level_paths = {}
                for tier in self.tiers[level]:
                    path = self.data_dir + self.fileDB.tier_dirs[tier] + '/' + self.fileDB.df.iloc[file][f'{tier}_file']
                    # Only add to level_paths if the file exists
                    if os.path.exists(path):
                        level_paths[tier] = path

                file_table = Table()
                for tb, idx in fdict[level].items():
                    level_table = None
                    for tier, path in level_paths.items():
                        template = self.fileDB.table_format[tier]
                        keyword = list(parse(template, template).named.keys())[0]
                        args = {keyword: tb}
                        tb_name = template.format(**args)
                        temp_tb, _ = sto.read_object(tb_name, path, idx=idx, field_mask=self.output_columns)
                        if level_table is None:
                            level_table = temp_tb 
                        else: 
                            level_table.join(temp_tb)
                    file_table.add_column(f"load{tb}", level_table, use_obj_size=True)
                if in_mem:
                    load_ret.append(file_table)
                if f_output:
                    fname = f_output + file
                    sto.write_object(file_table, f"file{file}", fname, wo_mode="overwrite_file")

            if in_mem: 
                if self.output_format == "lgdo.Table":
                    return load_ret
                elif self.output_format == "pd.DataFrame":
                    return [[t_out.get_dataframe() for t_out in f_tb.values()] for f_tb in load_ret]
                else:
                    print("I don't know how to output " + self.output_format + ", here is a lgdo.Table")
                    return load_ret
            
    def load_detector(self, det_id): #TODO
        """
        special version of `load` designed to retrieve all file files, tables,
        column names, and potentially calibration/dsp parameters relevant to one
        single detector.
        """
        pass

    def load_settings(self): #TODO
        """
        get metadata stored in raw files, usually from a DAQ machine.
        """
        pass

    def load_dsp_pars(self, query): #TODO
        """
        access the dsp_pars parameter database (probably JSON format) and do
        some kind of query to retrieve parameters of interest for our file list,
        and return some tables.
        """
        pass

    def load_cal_pars(self, query): #TODO
        """
        access the cal_pars parameter database, run a query, and return some tables.
        """
        pass

    def skim_waveforms(self, mode:str='hit', hit_list=None, evt_list=None): #TODO
            """
            handle this one separately because waveforms can easily fill up memory.
            """
            if mode=='hit':
                pass
            elif mode=='evt':
                pass
            pass

    def browse(self, query, dsp_config=None): #TODO
        """
        Interface between DataLoader and WaveformBrowser.
        """
        wb = WaveformBrowser()
        return wb

    def reset(self):
        self.file_list = None      
        self.table_list = None      
        self.cuts = None     
        self.merge_files = False
        self.output_format = 'lgdo.Table'
        self.output_columns = None 

class FileDB():
    """
    A class containing a pandas DataFrame that has additional functions to scan the data directory,
    fill the dataframe's columns with information about each file, and
    read/write to disk in an LGDO format
    """

    def __init__(self, config, file_df=None, scan=True):
        """
        Parameters
        ----------
            config : path to JSON file or dict
            Configuration file specifying data directories, tiers, and file name templates

            file_df : pd.DataFrame
            

            lgdo_file : string 
            Path to a file containing a LGDO.Table written out by FileDB.to_lgdo()

            scan : bool
            True by default, whether the fileDB should scan the DAQ directory to
            fill its rows with file information
        """
        if file_df is None:
            self.df = None 
            if isinstance(config, str):
                with open(config) as f:
                    config = json.load(f)                   

            self.set_config(config)

            # Set up column names
            fm = string.Formatter()
            parse_arr = np.array(list(fm.parse(self.file_format[self.tiers[0]])))
            names = list(parse_arr[:,1]) # fields required to generate file name
            names = [n for n in names if n] #Remove none values
            names = list(np.unique(names))
            names += [f'{tier}_file' for tier in self.tiers] # the generated file names
            names += [f'{tier}_size' for tier in self.tiers] # file sizes
            names += ['file_status', 'runtime'] # bonus columns 

            self.df = pd.DataFrame(columns=names)

            if scan:
                self.scan_files()
                self.set_file_status()
                self.set_file_sizes()
        else:
            self.from_disk(config, file_df)

    def set_config(self, config):
        self.config = config
        self.tiers = list(self.config["tier_dirs"].keys())
        self.file_format = self.config["file_format"]
        self.data_dir = self.config["data_dir"]
        self.tier_dirs = self.config["tier_dirs"]
        self.table_format = self.config["table_format"]

    def scan_files(self):
        """
        Scan the raw directory and fill the DataFrame
        Only fills columns that can be populated with just the DAQ file
        """
        file_keys = []
        n_files = 0
        low_tier = self.tiers[0]
        template = self.file_format[low_tier]
        scan_dir = self.data_dir + self.tier_dirs[low_tier]

        for path, folders, files in os.walk(scan_dir):
            n_files += len(files)

            for f in files:
                # in some cases, we need information from the path name
                if '/' in template:
                    f_tmp = path.replace(scan_dir,'') + '/' + f
                else:
                    f_tmp = f

                finfo = parse(template, f_tmp)
                if finfo is not None:
                    finfo = finfo.named
                    for tier in self.tiers:
                        finfo[f'{tier}_file'] = self.file_format[tier].format(**finfo)

                    file_keys.append(finfo)
                

        if n_files == 0:
            print(f"no {low_tier} files found...")
            return

        if len(file_keys) == 0:
            print(f"no {low_tier} files matched pattern", template)
            return

        temp_df = pd.DataFrame(file_keys)

        # fill the main DataFrame
        self.df = pd.concat([self.df, temp_df])

        # convert cols to numeric dtypes where possible
        for col in self.df.columns:
            try:
                self.df[col] = pd.to_numeric(self.df[col])
            except:
                pass
        
    def set_file_status(self):
        """
        Add a column to the dataframe with a bit corresponding to whether each tier's file exists
        e.g. if we have tiers "raw", "dsp", and "hit", but only the "raw" file has been made
                    file_status
        file1       0b100
        """
        def check_status(row):
            status = 0
            for i, tier in enumerate(self.tiers):
                path_name = self.data_dir + self.tier_dirs[tier] + '/' + row[f'{tier}_file']
                if os.path.exists(path_name):
                    status |= 1 << len(self.tiers)-i-1

            return status
        self.df['file_status'] = self.df.apply(check_status, axis=1)

    def set_file_sizes(self):
        def get_size(row, tier):
            size = 0
            path_name = self.data_dir + self.tier_dirs[tier] + '/' + row[f'{tier}_file']
            if os.path.exists(path_name):
                size = os.path.getsize(path_name)
            return size

        for tier in self.tiers:
            self.df[f'{tier}_size'] = self.df.apply(get_size, axis=1, tier=tier)

    def show(self, col_names:list=None):
        """
        show the existing fileDB as a DataFrame, optionally specifying columns
        """
        if col_names is None:
            print(self.df)
        else:
            print(self.df[col_names])

    def get_table_names(self):
        """
        Adds the available channels in each tier as a column in fileDB
        by searching for key names that match the provided table_format
        and saving the associated keyword values
                "raw_tables"            "evt_tables"
        file1   [0, 1, ...]     ["tcm", "grp_name", ...]
        """
               
        def update_table_names(row, tier):
            fpath = self.data_dir + self.tier_dirs[tier] + "/" + row[f'{tier}_file']

            try:
                f = h5py.File(fpath)
            except:
                return

            tier_tables = []
            template = self.table_format[tier]
            slashes = list(re.finditer('/', template))
            braces = list(re.finditer('{|}', template))

            if len(braces) > 2:
                print("Tables can only have one identifier")
            if len(braces)%2 != 0:
                print("Braces mismatch in table format")

             # Only need index of matches
            braces = [b.span()[0] for b in braces]
            slashes = [s.span()[0] for s in slashes]

            left_i = 0
            right_i = len(template)
            for s in slashes:
                if s < braces[0]:
                    left_i = s 
                if s > braces[1] and s < right_i:
                    right_i = s

            left = template[:left_i] 
            right = template[right_i+1:]
            mid = template[left_i:right_i] 

            if left:
                group = f[left]
            else:
                group = f 
            
            for key in group.keys():
                par_res = parse(mid, key)
                keys = list(par_res.named.keys())
            
                if par_res is not None:
                    tb = par_res.named[keys[0]]
                    if f"{left}/{key}/{right}" in f:
                        tier_tables.append(tb)

            return tier_tables 
            # End update_table_names

        t = time.perf_counter()
        for tier in self.tiers:
            self.df[f'{tier}_tables'] = self.df.apply(update_table_names, axis=1, tier=tier)
        print("Time: ", time.perf_counter()-t)

    def get_col_names(self, f_output:str=None):
        """
        Requires the {tier}_table columns of the dataframe to be filled, i.e. by running get_table_names()
        
        Returns a table with each unique list of columns found in each table
        Adds a column to the FileDB dataframe df['column_type'] that maps to the column table

        Optionally write the column table to LH5 file as a VectorOfVectors
        """
        def col_indices(row, tier):
            fpath = self.data_dir + self.tier_dirs[tier] + "/" + row[f'{tier}_file']
            col_idx = []
            try:
                f = h5py.File(fpath)
                for tb in row[f'{tier}_tables']:
                    template = self.table_format[tier]
                    keyword = list(parse(template, template).named.keys())[0]
                    args = {keyword: tb}
                    tb_name = template.format(**args)
                    col = list(f[tb_name].keys())
                    if col not in columns:
                        columns.append(col)
                        col_idx.append(len(columns)-1)
                    else:
                        col_idx.append(columns.index(col))
            except KeyError:
                print(f"Need \'{tier}_tables\' to get column names")
            except Exception as e:
                if tier == "raw":
                    print(e)
            return col_idx

        columns = []
        for tier in self.tiers:
            self.df[f'{tier}_col_idx'] = self.df.apply(col_indices, axis=1, tier=tier)
        
            
        if f_output is not None:
            flattened = []
            length = []
            for i, col in enumerate(columns):
                if i == 0:
                    length.append(len(col))
                else:
                    length.append(length[i-1]+len(col))
                for c in col:
                    flattened.append(c)
            columns_vov = VectorOfVectors(flattened_data=flattened, cumulative_length=length)
            sto = LH5Store()
            sto.write_object(columns_vov, 'unique_columns', f_output)
            
        return columns

    def from_disk(self, cfg_name, df_name):
        """
        Fills self.df and config with the information from a file created by to_lgdo()
        """
        with open(cfg_name, "r") as cfg:
            config = json.load(cfg)
        self.set_config(config)
        self.df = pd.read_hdf(df_name, key="file_df")

    def to_disk(self, cfg_name, df_name):
        """
        Writes config information to cfg_name and DataFrame to df_name

        cfg_name should be a JSON file
        df_name should be an HDF5 file

        Parameters
        -----------
            cfg_name : string
            Path to output file for config

            df_name : string
            Path to output file for DataFrame
        Returns
        -------
            None. 
        """
        with open(cfg_name, "w") as cfg:
            json.dump(self.config, cfg)

        self.df.to_hdf(df_name, "file_df")
        
    def scan_daq_files(self, verbose=False):
        """
        Does the exact same thing as scan_files but with extra config arguments for a DAQ directory and template
        instead of using the lowest (raw) tier 
        """
        file_keys = []
        n_files = 0

        for path, folders, files in os.walk(self.daq_dir):
            n_files += len(files)

            for f in files:

                # in some cases, we need information from the path name
                if '/' in self.daq_template:
                    f_tmp = path.replace(self.daq_dir,'') + '/' + f
                else:
                    f_tmp = f

                finfo = parse(self.daq_template, f_tmp)
                if finfo is not None:
                    finfo = finfo.named
                    file_keys.append(finfo)
                for tier in self.tiers:
                    finfo[f'{tier}_file'] = self.file_format[tier].format(**finfo)

                


        if n_files == 0:
            print("no daq files found...")
            return

        if len(file_keys) == 0:
            print("no daq files matched pattern", self.daq_template)
            return

        temp_df = pd.DataFrame(file_keys)

        # fill the main DataFrame
        self.df = pd.concat([self.df, temp_df])

        # convert cols to numeric dtypes where possible
        for col in self.df.columns:
            try:
                self.df[col] = pd.to_numeric(self.df[col])
            except:
                pass
        
        if verbose:
            print(self)

if __name__=='__main__':
    doc="""
    Demonstrate usage of the `DataLoader` class.
    This could be what we initially run at LNGS - it would try to do the `os.walk`
    method over the existing files, and e.g. scan for existence of various files
    in different stages.  More advanced tests would be moved to a notebook or
    separate script.
    """
    def pretty_print_dict(d, indent=0):
        for key, value in d.items():
            if isinstance(value, dict):
                print('\t' * indent + str(key))
                pretty_print_dict(value, indent+1)
            else:
                print('\t' * indent + str(key) + ":\t" + str(value))

    print('-------------------------------------------------------')

    print("Full FileDB: ")
    dl = DataLoader(config="../../../../loader_config.json", 
                    fileDB_config="../../../../fileDB_config.json")
    dl.show_fileDB()
    print()

    # print("Read/write FileDB: ")
    # dl.fileDB.to_disk("fileDB_cfg.json", "fileDB_df.h5")
    # dl2 = DataLoader(config="../../../../loader_config.json", fileDB_config="fileDB_cfg.json", fileDB="fileDB_df.h5")
    # dl2.show_fileDB()

    print("Get table names: ")
    dl.fileDB.get_table_names()
    dl.show_fileDB()
    print()

    print("Files where timestamp >= 20230101T0000")
    dl.set_files("timestamp >= '20230101T0000'")
    dl.show_file_list()
    print()

    print("Files where timestamp == 20220629T003837")
    dl.set_files("timestamp == '20220629T003837'")
    dl.show_file_list(columns=["raw_file", "file_status", "tcm_file"])
    print()

    print("Get Columns: ")
    cols = dl.fileDB.get_col_names()
    dl.show_fileDB(['raw_tables', 'raw_col_idx'])
    print(cols)
    print()

    print("Set cuts and get entries: ")
    dl.set_cuts({"hit": "timestamp > "})
    el = dl.gen_entry_list() 
    pretty_print_dict(el)
    print()

    print("Load data, merge, Tables: ")
    cols = ["daqenergy", "waveform"]
    dl.set_output(fmt="lgdo.Table", merge_files=True, columns=cols)
    lout = dl.load(el, f_output="test_load.lh5", in_mem=True)
    pretty_print_dict(lout)
    print()

    print("Load data, no merge, DataFrames: ")
    cols = ["daqenergy", "timestamp"]
    dl.set_output(fmt="pd.DataFrame", merge_files=False, columns=cols)
    lout = dl.load(el, in_mem=True)
    print(len(lout))
    print(len(lout[0]))
    for f, ftb in enumerate(lout):
        print(f"File {f}")
        for df in ftb:
            print(df)
    print('-------------------------------------------------------')