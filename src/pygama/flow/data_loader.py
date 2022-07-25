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
from pygama.flow.fileDB import FileDB
from pprint import pprint
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
        self.evts = {}
        self.tcms = {}
        for level in self.levels:
            self.tiers[level] = config["levels"][level]["tiers"]
            #Set cut priority
            if "parent" in config["levels"][level].keys(): # This level is a TCM
                parent = config["levels"][level]["parent"]
                child = config["levels"][level]["child"]
                self.tcms[level] = config["levels"][level] 
                self.cut_priority[level] = self.cut_priority[parent] + 2
                self.cut_priority[child] = self.cut_priority[parent] + 1
                self.evts[child] = {"tcm": level, "parent": parent}

                #Set TCM columns to lookup
                if "tcm_cols" not in config["levels"][level].keys():
                    print(f"Config Warning: TCM levels, e.g. {level}, need to specify the TCM lookup columns")
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

    def get_table_name(self, tier, tb):
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
        return table_name

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

    def get_tiers_for_col(self, columns, merge_files=None):
        """
        Get the tiers, and tables in that tier, that contain the columns given

        col_tiers = {
            file: {
                "tables": {
                    "raw": [0, 1, 2, 3],
                    "dsp": [0, 1, 2, 3],
                    "tcm": [""]
                },
                "columns": {
                    "daqenergy": "raw",
                    "trapEmax": "dsp",
                    .
                    .
                    .
                }
            }
        }
        """
        col_tiers = {}
        
        if merge_files is None:
            merge_files = self.merge_files

        if merge_files:
            for file in self.file_list:
                col_inds = set()
                for i, col_list in enumerate(self.fileDB.columns):
                    if not set(col_list).isdisjoint(columns):
                        col_inds.add(i)

                for level in self.levels:
                    for tier in self.tiers[level]:
                        col_tiers[tier] = set()
                        if self.fileDB.df.loc[file,f"{tier}_col_idx"] is not None:
                            for i in range(len(self.fileDB.df.loc[file,f"{tier}_col_idx"])):
                                if self.fileDB.df.loc[file, f"{tier}_col_idx"][i] in col_inds:
                                    col_tiers[tier].add(self.fileDB.df.loc[file, f"{tier}_tables"][i]) 
        else:
            for file in self.file_list:
                col_tiers[file] = {
                    "tables": {},
                    "columns": {}
                }
                col_inds = set()
                for i, col_list in enumerate(self.fileDB.columns):
                    if not set(col_list).isdisjoint(columns):
                        col_inds.add(i)

                for level in self.levels:
                    for tier in self.tiers[level]:
                        col_tiers[file]["tables"][tier] = []
                        if self.fileDB.df.loc[file,f"{tier}_col_idx"] is not None:
                            for i in range(len(self.fileDB.df.loc[file,f"{tier}_col_idx"])):
                                col_idx = self.fileDB.df.loc[file, f"{tier}_col_idx"][i]
                                if col_idx in col_inds:
                                    col_tiers[file]["tables"][tier].append(self.fileDB.df.loc[file, f"{tier}_tables"][i]) 
                                    col_in_tier = set.intersection(set(self.fileDB.columns[col_idx]), set(columns))
                                    for c in col_in_tier:
                                        col_tiers[file]["columns"][c] = tier
                        
        return col_tiers 

    def gen_entry_list(self, tcm_level=None, tcm_table=None, chunk=False, mode='only', save_output_columns=False, in_mem=True, f_output=None): #TODO: mode, chunking, etc
        """
        This should apply cuts to the tables and files of interest
        but it does NOT load the column information into memory

        Can only load up to two levels, those joined by tcm_level
        
        Parameters
        ----------
        tcm_level : string 
        The type of tcm to be used
        If None, will only return information from lowest level
        
        tcm_table : int or string
        The identifier of the table inside this TCM level that you want to use
        If not specified, there must only be one table inside a TCM file in tcm_level


        chunk : bool ?????????????????
        If true, iterates through each file in file_list
        If false, opens all files at once 

        mode : 'any' or 'only'
        'any' : returns every hit in the event if any hit in the event passes the cuts
        'only' : only returns hits that pass the cuts

        save_output_columns : bool
        If true, saves any columns needed for both the cut and the output to the entry_list

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

        if not in_mem and f_output is None:
            print("Need to specify output file if not returning in memory")
            return

        if in_mem:
            entries = {}
        low_level = self.levels[0]

        # Default columns in the entry_list
        if tcm_level is None:
            return self.gen_hit_entries(save_output_columns, in_mem, f_output)
        else:
            # Assumes only one tier in a TCM level
            parent = self.tcms[tcm_level]["parent"]
            child = self.tcms[tcm_level]["child"]
            entry_cols = [f"{parent}_table", f"{parent}_idx", f"{child}_table", f"{child}_idx"]
            if save_output_columns:
                for_output = []

            # Find out which columns are needed for any cuts
            cut_cols = {}

            for level in [child, parent]:
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
                            if term in self.output_columns and term not in entry_cols and save_output_columns:
                                for_output.append(term)
            if save_output_columns:
                entry_cols += for_output
            sto = LH5Store()

            # Make the entry list for each file
            for file in self.file_list:
                # Get the TCM specified
                tcm_tier = self.tiers[tcm_level][0] # Assumes that each TCM level only has one tier
                tcm_tables = self.fileDB.df.loc[file,f"{tcm_tier}_tables"]
                if len(tcm_tables) > 1 and tcm_table is None:
                    print(f"There are {len(tcm_tables)} TCM tables, need to specify which to use")
                    print(tcm_tables)
                    return 
                else:
                    if tcm_table is not None:
                        if tcm_table in tcm_tables:
                            tcm_tb = tcm_table 
                        else:
                            print(f"Table {tcm_table} doesn't exist in {tcm_level}")
                            return
                    else:
                        tcm_tb = tcm_tables[0]
                tcm_path = self.data_dir + self.fileDB.tier_dirs[tcm_tier] + '/' + self.fileDB.df.iloc[file][f'{tcm_tier}_file']
                if not os.path.exists(tcm_path):
                    print(f"Can't find TCM file for {tcm_level}")
                    return 
                tcm_table_name = self.get_table_name(tcm_tier, tcm_tb)
                tcm_lgdo, _ = sto.read_object(tcm_table_name, tcm_path) 
                f_entries = tcm_lgdo.get_dataframe() 
                renaming = {
                    self.tcms[tcm_level]["tcm_cols"]["child_idx"]: f"{child}_idx", 
                    self.tcms[tcm_level]["tcm_cols"]["parent_tb"]: f"{parent}_table",
                    self.tcms[tcm_level]["tcm_cols"]["parent_idx"]: f"{parent}_idx"
                }
                f_entries.rename(columns=renaming, inplace=True)
                # At this point, should have a list of all available hits/evts joined by tcm

                # Perform cuts specified for child or parent level, in that order
                for level in [child, parent]:
                    if level not in self.cuts.keys():
                        continue
                    cut = self.cuts[level]
                    col_tiers = self.get_tiers_for_col(cut_cols[level], merge_files=False)
                    # Tables in first tier of event should be the same for all tiers in one level
                    tables = self.fileDB.df.loc[file, f'{self.tiers[level][0]}_tables'] 
                    if self.table_list is not None:
                        if level in self.table_list.keys():
                            tables = self.table_list[level]
                    # Cut any rows of TCM not relating to requested tables
                    f_entries.query(f"{level}_table in {tables}", inplace=True) 

                    for tb in tables:
                        tb_table = None 
                        tcm_idx = f_entries.query(f"{level}_table == {tb}").index 
                        idx_mask = f_entries.loc[tcm_idx, f"{level}_idx"]
                        for tier in self.tiers[level]:
                            tier_path = self.data_dir + self.fileDB.tier_dirs[tier] + '/' + self.fileDB.df.loc[file,f'{tier}_file']
                            if tier in col_tiers[file]["tables"].keys():
                                if tb in col_tiers[file]["tables"][tier]:
                                    table_name = self.get_table_name(tier, tb)
                                    tier_table, _ = sto.read_object(table_name, tier_path, field_mask=cut_cols[level], idx=list(idx_mask))
                                    if tb_table is None:
                                        tb_table = tier_table 
                                    else:
                                        tb_table.join(tier_table) 
                        tb_df = tb_table.get_dataframe()
                        tb_df.query(cut, inplace=True) 
                        keep_idx = f_entries.query(f"{level}_table == {tb} and {level}_idx in {list(tb_df.index)}").index
                        drop_idx = set.symmetric_difference(set(tcm_idx), list(keep_idx))
                        f_entries.drop(drop_idx, inplace=True)
                        if save_output_columns:
                            for col in tb_df.columns:
                                if col in for_output:
                                    f_entries.loc[keep_idx, col] = tb_df[col].tolist()
                        #end for each table loop
                    # end for each level loop

                if in_mem:
                    entries[file] = f_entries
                if f_output:
                    # Convert f_entries DataFrame to Struct
                    f_dict = f_entries.to_dict('list')
                    f_struct = Struct(f_dict)
                    sto.write_object(f_struct, f"entries/{file}", f_output, wo_mode='o')
                #end for each file loop
        if in_mem:
            return entries

    def gen_hit_entries(self, save_output_columns=False, in_mem=True, f_output=None):
        """
        Called by gen_entry_list() to handle the case when tcm_level is None
        Ignores any cuts set on levels above lowest level
        """
        low_level = self.levels[0]
        if in_mem:
            entries = {}
        entry_cols = [f"{low_level}_table", f"{low_level}_idx"]

        # Find out which columns are needed for the cut
        cut_cols = []
        cut = ""
        if self.cuts is not None and low_level in self.cuts.keys():
            cut = self.cuts[low_level] 
            # String parsing to determine which columns need to be loaded
            terms = re.findall('[^\W0-9]\w*', cut) # Matches any valid python variable name
            for term in terms:
                if not iskeyword(term):
                    cut_cols.append(term)
                    # Add column to entry_cols if they are needed for both the cut and the output
                    if self.output_columns is not None:
                        if term in self.output_columns and term not in entry_cols and save_output_columns:
                            entry_cols.append(term)

        col_tiers = self.get_tiers_for_col(cut_cols, merge_files=False)
        sto = LH5Store()
        for file in self.file_list:
            f_entries = pd.DataFrame(columns=entry_cols)
            if self.table_list is not None:
                if low_level in self.table_list.keys():
                    tables = self.table_list[low_level] 
            else:
                tables = self.fileDB.df.loc[file, f"{self.tiers[low_level][0]}_tables"]

            for tb in tables:
                tb_table = None
                if not cut:
                    tier = self.tiers[low_level][0]
                    tb_idx = self.fileDB.df.loc[file, f"{tier}_tables"].index(tb)
                    columns = self.fileDB.columns[self.fileDB.df.loc[file, f"{tier}_col_idx"][tb_idx]]
                    tier_path = self.data_dir + self.fileDB.tier_dirs[tier] + '/' + self.fileDB.df.iloc[file][f'{tier}_file']
                    if os.path.exists(tier_path):
                        table_name = self.get_table_name(tier, tb)
                        tb_table, _ = sto.read_object(table_name, tier_path, field_mask=[columns[0]])
                else:
                    for tier in self.tiers[low_level]:
                        if tier in col_tiers[file]["tables"].keys():
                            tier_path = self.data_dir + self.fileDB.tier_dirs[tier] + '/' + self.fileDB.df.iloc[file][f'{tier}_file']
                            if tier in col_tiers[file]["tables"].keys():
                                if tb in col_tiers[file]["tables"][tier]:
                                    table_name = self.get_table_name(tier, tb)
                                    tier_tb, _ = sto.read_object(table_name, tier_path, field_mask=cut_cols)
                                    if tb_table is None:
                                        tb_table = tier_tb 
                                    else:
                                        tb_table.join(tier_tb) 
                tb_df = tb_table.get_dataframe()
                if cut:
                    tb_df.query(cut, inplace=True)
                tb_df[f"{low_level}_table"] = tb
                tb_df[f"{low_level}_idx"] = tb_df.index
                f_entries = pd.concat((f_entries, tb_df), ignore_index=True)[entry_cols]
            if in_mem:
                entries[file] = f_entries 
            if f_output:
                # Convert f_entries DataFrame to Struct
                f_dict = f_entries.to_dict('list')
                f_struct = Struct(f_dict)
                sto.write_object(f_struct, f"entries/{file}", f_output, wo_mode='o')
        if in_mem:
            return entries

    def load(self, entry_list=None, in_mem=False, f_output=None, rows='hit', tcm_level=None): #TODO
        if entry_list is None:
            print("First run gen_entry_list and pass the output to load")
            return 

        if in_mem == False and f_output is None:
            print("If in_mem is False, need to specify an output file")
            return

        if rows == 'hit':
            return self.load_hits(entry_list, in_mem, f_output)
        elif rows == 'evt':
            if tcm_level is None:
                print("Need to specify which coincidence map to use to return event-oriented data")
                return
            return self.load_evts(entry_list, in_mem, f_output, tcm_level)
        else:
            print(f"I don't understand what rows={rows} means!")
            return

    def load_hits(self, entry_list=None, in_mem=False, f_output=None):
        """
        Actually retrieve the information from the events in entry_list, and 
        return it in the requested output format 
        """
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
                        # Find which tiers we need to load
                        col_tiers = self.get_tiers_for_col(self.output_columns, merge_files=True)
                        if tb not in col_tiers[tier]:
                            continue

                        # Get table name
                        table_name = self.get_table_name(tier, tb)

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
                        # print("level: ", level)
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
                            # print("level paths found")
                            for tier, path in level_paths.items():
                                col_tiers = self.get_tiers_for_col(columns=self.output_columns)
                                if level == low_level:
                                    tb_id = tb
                                else:
                                    tb_id = ""
                                    
                                if tb_id not in col_tiers[file][tier]:
                                    continue

                                # Get table name for tier
                                table_name = self.get_table_name(tier, tb_id)
                                
                                # Load tier
                                temp_tb, _ = sto.read_object(table_name, path, idx=idx, field_mask=self.output_columns)

                                if level_table is None:
                                    level_table = temp_tb 
                                else:
                                    level_table.join(temp_tb) 

                        if level_table is not None:
                            if level in self.tcms:
                                level_df = level_table.get_dataframe()
                                tb_df = level_df.query( f"{self.tcm_cols[level]['hit_tb']} == {tb}" ).sort_values(self.tcm_cols[level]["hit_idx"])
                                level_cols = {}
                                for col in level_table.keys():
                                    nda = []
                                    for i in range(tb_df[self.tcm_cols[level]["hit_idx"]].iloc[-1]):
                                        if i in tb_df[self.tcm_cols[level]["hit_idx"]]:
                                            nda.append(tb_df[col].iloc[i])
                                        else:
                                            nda.append(np.nan)
                                    level_cols[col] = Array(nda=np.array(nda))
                            # print("level table found")
                            # if self.cut_priority[level] > 0:
                            #     level_table = level_table.explode(list(self.tcm_cols.values()))

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
            
    def load_evts(self, entry_list=None, in_mem=False, f_output=None, tcm_level=None):
        sto = LH5Store()
        self.output_columns += list(self.tcm_cols[tcm_level].values())
        print(self.output_columns)
        col_tiers = self.get_tiers_for_col(self.output_columns)
        if self.merge_files:
            pass
        else: # End merge_files 
            load_out = {}
            for file, f_entries in entry_list:
                # Pre-allocate memory for each output column
                evt_len = len(f_entries['cumulative_length'])
                hit_len = len(f_entries[f'{self.levels[0]}_idx'])
                flattened_out = {}
                for col in self.output_columns:
                    col_tier = col_tiers[file]["columns"][col]
                    for level in self.levels:
                        if col_tier in self.tiers[level]:
                            col_level = level 
                    if self.cut_priority[level] > 0:
                        flattened_out[col] = np.empty(evt_len)
                    else:
                        flattend_out[col] = np.empty(hit_len)

                for level in self.levels:
                    for tb in set(f_entries[f"{level}_table"]):
                        if self.cut_priority[level] == 0:
                            tcm_idx = np.where(f_entries[f"{level}_table"] == tb)[0]
                            idx_mask = f_entries[f"{level}_idx"][tcm_idx] 
                            for tier in self.tiers[level]:
                                if tb in col_tiers[file]["tables"][tier]:
                                    table_name = get_table_name(tier, tb)
                                    path = self.data_dir + self.fileDB.tier_dirs[tier] + "/" + self.fileDB.df.loc[file, f"{tier}_path"]
                                    temp_table, _ = sto.read_object(table_name, path, idx=idx_mask, field_mask=self.output_columns) 
                                    for col in temp_table.keys():
                                        flattened_output[col][tcm_idx] = temp_table[col] 

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


if __name__=='__main__':
    doc="""
    Demonstrate usage of the `DataLoader` class.
    This could be what we initially run at LNGS - it would try to do the `os.walk`
    method over the existing files, and e.g. scan for existence of various files
    in different stages.  More advanced tests would be moved to a notebook or
    separate script.
    """

    print('-------------------------------------------------------')

    print("Full FileDB: ")
    dl = DataLoader(config="../../../../loader_config.json", 
                    fileDB_config="../../../../fileDB_config_copy.json")
    dl.show_fileDB()
    print()

    print("Get table and column names:")
    pd.set_option('display.max_colwidth', 10)
    dl.fileDB.get_tables_columns()
    dl.show_fileDB()
    print()

    # print("Read/write FileDB: ")
    # dl.fileDB.to_disk("fileDB_cfg.json", "fileDB_df.h5")
    # dl2 = DataLoader(config="../../../../loader_config.json", fileDB_config="fileDB_cfg.json", fileDB="fileDB_df.h5")
    # dl2.show_fileDB()


    print("Files where timestamp >= 20230101T0000Z")
    dl.set_files("timestamp >= '20230101T0000Z'")
    dl.show_file_list()
    print()

    print("Files where timestamp == '20220628T233014Z'")
    dl.set_files("timestamp == '20220628T233014Z'")
    dl.show_file_list(columns=["raw_file", "file_status", "tcm_file"])
    print()

    dl.set_datastreams([1, 42], "ch")
    print(dl.table_list)
    print()

    print("Set cuts and get entries: ")
    hit_el = dl.gen_entry_list() 
    print("No cuts, no arguments:")
    pprint(hit_el) 
    dl.set_cuts({"hit": "daqenergy > 0"})
    print("Energy cut, no arguments")
    hit_cut_el = dl.gen_entry_list()
    pprint(hit_cut_el)
    print("Energy cut, tcm")
    tcm_el = dl.gen_entry_list(tcm_level="tcm")
    pprint(tcm_el)
    print()

    print("Save output columns to entry list: ")
    cols = ["daqenergy", "tp_20"]
    dl.set_output(fmt="lgdo.Table", merge_files=False, columns=cols)
    dl.set_cuts({"hit": "daqenergy > 0"})
    print("Energy cut, no arguments")
    hit_cut_el = dl.gen_entry_list(save_output_columns=True)
    pprint(hit_cut_el)
    print("Energy cut, tcm")
    tcm_el = dl.gen_entry_list(tcm_level="tcm", save_output_columns=True)
    pprint(tcm_el)
    print()

    # print("Load data, merge, Tables: ")    
    # dl.set_output(fmt="lgdo.Table", merge_files=True, columns=cols)
    # lout = dl.load(el, in_mem=True)
    # pretty_print_dict(lout)
    # print()

    print("Load data, no merge: ")
    lout = dl.load(el, in_mem=True, rows="evt", tcm_level="tcm")
    pprint(lout)
    print('-------------------------------------------------------')