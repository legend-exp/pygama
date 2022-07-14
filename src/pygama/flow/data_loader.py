import os
import json
import pandas as pd
import string
import re
import numpy as np
import h5py
import time
from keyword import iskeyword
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
            
            template = self.fileDB.table_format[tier] 
            fm = string.Formatter()
            parse_arr = np.array(list(fm.parse(template)))
            names = list(parse_arr[:,1]) # fields required to generate file name
            if len(names) > 0:
                keyword = names[0]

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

    def gen_entry_list(self, chunk=False, mode='only', f_output=None): #TODO: mode, chunking, etc
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

        # Columns to save because we know the user wants them in the final output
        for_output = []

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
                if term.isidentifier() and not iskeyword(term): #Assumes that column names are valid python variable names
                    cut_cols[level].append(term)
                    # Add column to entry_cols if they are needed for both the cut and the output
                    if self.output_columns is not None:
                        if term in self.output_columns and term not in entry_cols:
                            for_output.append(term)
        
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
                    fields = []
                    # Use tcm_cols to find column names that correspond to level_table and level_idx 
                    if level in self.tcm_cols.keys():
                        fields = list(self.tcm_cols[level].values())
                    
                    columns = entry_cols + fields + cut_cols[level]

                    sto = LH5Store()
                    for tb in tables:
                        # Try to create index mask for this table
                        if first_cut_made:
                            idx = list(f_entries.query(f"{level}_table == {tb}")[f"{level}_idx"])
                        else:
                            idx = None
                        if idx == []:
                            continue
                        level_table = None 
                        for tier, path in level_paths.items():
                            template = self.fileDB.table_format[tier]
                            fm = string.Formatter()
                            parse_arr = np.array(list(fm.parse(template)))
                            names = list(parse_arr[:,1])
                            if len(names) > 0:
                                keyword = names[0]
                                args = {keyword: tb}
                                table_name = template.format(**args)
                            else:
                                table_name = template
                            temp_table, _ = sto.read_object(table_name, path, idx=idx, field_mask=columns)
                            if level_table is None:
                                level_table = temp_table
                            else:
                                level_table.join(temp_table) 
                        level_df = level_table.get_dataframe()
                        cut_df = level_df.query(cut)
                        # Rename columns to match entry_cols
                        if level in self.tcm_cols.keys():
                            dep = self.config["levels"][level]["dependency"]
                            renaming = {
                                self.tcm_cols[level]["self_idx"]: f"{level}_idx",
                                self.tcm_cols[level]["table_col"]: f"{dep}_table", 
                                self.tcm_cols[level]["idx_col"]: f"{dep}_idx", 
                            }
                            cut_df = cut_df.rename(renaming, axis="columns")
                            cut_df = cut_df.explode(list(renaming.values()), ignore_index=True)
                        else:
                            cut_df.loc[:,f"{level}_idx"] = cut_df.index
                            cut_df.loc[:,f"{level}_table"] = [tb]*len(cut_df)

                        # Update the entry list with latest cut
                        if not first_cut_made:
                            f_entries = pd.concat((f_entries, cut_df), ignore_index=True)[entry_cols]
                        else:
                            tb_entries = f_entries.query(f"{level}_table == {tb}")
                            drop_entries = tb_entries.query(f"{level}_idx not in {list(cut_df.index)}")
                            f_entries = f_entries.drop(list(drop_entries.index))
                        f_entries = f_entries.reset_index(drop=True)
                        #end for each table loop

                first_cut_made = True
                # end for each level loop
                            
            entries[file] = f_entries
            #end for each file loop


        if f_output is not None:
            sto = LH5Store()
            # Convert entry dataframe to lgdo.Table to write to disk
            # Can do this because we know each column of the entry list is just a list of scalars
            for file, entry_df in entries.items():
                col_dict = {}
                for col in entry_df.columns:
                    arr = Array(nda=np.array(entry_df[col]))
                    col_dict[col] = arr 
                entry_tb = Table(col_dict)
                sto.write_object(entry_tb, f"entries{file}", f_output)
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

        low_level = self.levels[0]

        sto = LH5Store()
        writing = False
        if self.merge_files: # Try to load all information at once
            load_out = Struct()


            load_tbs = []
            
            #Get all tables we may be interested in
            for file in entry_list.keys():
                f_tbs = entry_list[file][f"{low_level}_table"].unique()
                for tb in f_tbs:
                    if tb not in load_tbs:
                        load_tbs.append(tb) 

            for tb in load_tbs:
                tb_table = None
                for level in self.levels:
                    level_table = None
                    for tier in self.tiers[level]:
                        # Determine if tier needs to be loaded 
                        if level == low_level:
                            tb_idx = []
                            for i in range(len(entry_list)):
                                if list((self.fileDB.df[f"{tier}_tables"].loc[entry_list.keys()]))[i] is None:
                                    continue
                                tb_idx.append(list(self.fileDB.df[f"{tier}_tables"][entry_list.keys()])[i].index(tb))
                        else:
                            tb_idx = [0]*len(entry_list) # Assumes that higher levels only have one table per file
                        if not tb_idx:
                            print(f"Skipping {tier}")
                            continue
                        else:
                            print(f"Keeping {tier}")
                        col_idx = []
                        for i, row in enumerate(self.fileDB.df[f"{tier}_col_idx"].loc[entry_list.keys()]):
                            if row:
                                if row[tb_idx[i]] not in col_idx:
                                    col_idx.append(row[tb_idx[i]])

                        tier_cols = [] 
                        for idx in col_idx:
                            tier_cols += self.fileDB.columns[idx]
                            
                        if set(self.output_columns).isdisjoint(tier_cols):
                            continue

                        # Get table name
                        template = self.fileDB.table_format[tier]
                        fm = string.Formatter()
                        parse_arr = np.array(list(fm.parse(template)))
                        names = list(parse_arr[:,1]) 
                        if len(names) > 0:
                            keyword = names[0]
                            args = {keyword: tb}
                            table_name = template.format(**args)
                        else:
                            table_name = template

                        # Get file paths
                        paths = [ self.data_dir + self.fileDB.tier_dirs[tier] + '/' + self.fileDB.df.iloc[file][f'{tier}_file'] for file in entry_list.keys()]
                        p_exists = [os.path.exists(p) for p in paths] 
                        paths = [p for p, t in zip(paths, p_exists) if t] 

                        # Get index lists
                        inds = [ entry_list[file].query(f"{level}_table == {tb}")[f"{level}_idx"].to_list() for file in entry_list.keys()]

                        if paths:
                            sto = LH5Store()
                            tier_table, _ = sto.read_object(table_name, paths, idx=inds, field_mask=self.output_columns)                             
                            if level_table is None:
                                level_table = tier_table 
                            else:
                                print("level joining tier")
                                print("level_table", level_table) 
                                print("tier_table", tier_table)
                                level_table.join(tier_table) 
                    if level_table is not None:
                        if self.cut_priority[level] > 0:
                            level_table = level_table.explode(list(self.tcm_cols[level].values()))

                        if tb_table is None:
                            tb_table = level_table 
                        else:
                            print("tb joining level")
                            tb_table.join(level_table)
                    load_out[tb] = tb_table
                

            if in_mem:
                if self.output_format == "lgdo.Table":
                        return load_out
                elif self.output_format == "pd.DataFrame":
                    return [tb.get_dataframe() for tb in load_out.values()]
                else:
                    print("I don't know how to output " + self.output_format + ", here is a lgdo.Table")
                    return load_out
            else:
                return
        else: #Not merge_files
            load_out = {}
            
            for file, f_entries in entry_list.items():
                f_struct = Struct()
                for tb in f_entries[f"{low_level}_table"].unique(): # Assumes that higher levels only have one table per file
                    tb_table = None
                    for level in self.levels:
                        level_table = None

                        # Get valid file paths
                        level_paths = {}
                        for tier in self.tiers[level]:
                            path = self.data_dir + self.fileDB.tier_dirs[tier] + '/' + self.fileDB.df.iloc[file][f'{tier}_file']
                            # Only add to level_paths if the file exists
                            if os.path.exists(path):
                                level_paths[tier] = path
                        
                        # Make index mask
                        idx = list(f_entries.query(f"{low_level}_table == {tb}")[f"{level}_idx"])

                        if level_paths:
                            for tier, path in level_paths.items():
                                # Determine if tier needs to be loaded 
                                if level == low_level:
                                    tb_idx = list(self.fileDB.df[f"{tier}_tables"][file]).index(tb)
                                else:
                                    tb_idx = 0 # Assumes that higher levels only have one table per file
                                tier_cols = self.fileDB.columns[ self.fileDB.df[f"{tier}_col_idx"][file][tb_idx] ]
                                if set(self.output_columns).isdisjoint(tier_cols):
                                    continue

                                # Get table name for tier
                                template = self.fileDB.table_format[tier]
                                fm = string.Formatter()
                                parse_arr = np.array(list(fm.parse(template)))
                                names = list(parse_arr[:,1]) 
                                if len(names) > 0:
                                    keyword = names[0]
                                    args = {keyword: tb}
                                    table_name = template.format(**args)
                                else:
                                    table_name = template
                                
                                # Load tier
                                temp_tb, _ = sto.read_object(table_name, path, idx=idx, field_mask=self.output_columns)

                                if level_table is None:
                                    level_table = temp_tb 
                                else:
                                    level_table.join(temp_tb) 

                        if level_table is not None:
                            if self.cut_priority[level] > 0:
                                level_table = level_table.explode(list(self.tcm_cols.values()))

                            if tb_table is None:
                                tb_table = level_table 
                            else:
                                tb_table.join(level_table) 
                    f_struct[tb] = tb_table

                load_out[file] = f_struct
                if f_output:
                    fname = f_output + file
                    sto.write_object(f_struct, f"file{file}", fname, wo_mode="overwrite_file")

            if in_mem: 
                if self.output_format == "lgdo.Table":
                    return load_out
                elif self.output_format == "pd.DataFrame":
                    return [[t_out.get_dataframe() for t_out in f_tb.values()] for f_tb in load_out]
                else:
                    print("I don't know how to output " + self.output_format + ", here is a lgdo.Table")
                    return load_out
            
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
            if len(braces) == 0:
                tier_tables.append(template)
                return tier_tables

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

            print(left + ", " + mid + ", " + right)
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
                    fm = string.Formatter()
                    parse_arr = np.array(list(fm.parse(template)))
                    names = list(parse_arr[:,1]) # fields required to generate file name
                    if len(names) > 0:
                        keyword = names[0]
                        args = {keyword: tb}
                        table_name = template.format(**args)
                    else:
                        table_name = template
                    col = list(f[table_name].keys())
                    if col not in columns:
                        columns.append(col)
                        col_idx.append(len(columns)-1)
                    else:
                        col_idx.append(columns.index(col))
            except KeyError as e:
                print(f"Need \'{tier}_tables\' to get column names")
                print(e)
            except Exception as e:
                if tier == "raw" or tier=="tcm":
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
            
        self.columns = columns
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
                print('\t' * indent + str(key) + ":")
                print('\t' * indent + str(value))

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

    print("Files where timestamp >= 20230101T0000Z")
    dl.set_files("timestamp >= '20230101T0000Z'")
    dl.show_file_list()
    print()

    print("Files where timestamp >= 20220629T003746Z")
    dl.set_files("timestamp >= '20220629T003746Z'")
    dl.show_file_list(columns=["raw_file", "file_status", "tcm_file"])
    print()

    print("Get Columns: ")
    cols = dl.fileDB.get_col_names()
    dl.show_fileDB(['raw_tables', 'raw_col_idx', 'tcm_col_idx'])
    print(cols)
    print()

    dl.set_datastreams([0, 1, 42], "ch")
    print(dl.table_list)
    print()

    print("Set cuts and get entries: ")
    dl.set_cuts({"hit": "daqenergy > 0 and daqenergy < 1000"})
    el = dl.gen_entry_list() 
    pretty_print_dict(el) 
    print()

    cols = ["daqenergy", "timestamp"]
    print("Load data, merge, Tables: ")    
    dl.set_output(fmt="lgdo.Table", merge_files=True, columns=cols)
    lout = dl.load(el, in_mem=True)
    pretty_print_dict(lout)
    print()

    print("Load data, no merge: ")
    dl.set_output(fmt="lgdo.Table", merge_files=False, columns=cols)
    lout = dl.load(el, in_mem=True)
    pretty_print_dict(lout)
    print('-------------------------------------------------------')