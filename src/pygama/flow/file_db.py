from __future__ import annotations

import json
import logging
import os
import re
import string

import h5py
import numpy as np
import pandas as pd
from parse import parse

from pygama.lgdo import Array, LH5Store, VectorOfVectors
from pygama.lgdo.lh5_store import ls

log = logging.getLogger(__name__)


class FileDB:
    """
    A class containing a pandas DataFrame that has additional functions to scan the data directory,
    fill the dataframe's columns with information about each file, and
    read/write to disk in an LGDO format

    Columns
    -------
        file keys:
        The fields specified in the configuration file's "file_format" that are required to generate a file name
        e.g. "run", "type", "timestamp"

        {tier}_file:
        Generated file name for tier

        {tier}_size:
        Size of file on disk, if applicable

        file_status:
        Contains a bit corresponding to whether or not a file for each tier exists for a given cycle
        e.g. If we have tiers "raw", "dsp", and "hit", but only the "raw" file has been made
        file_status would be 0b100

        {tier}_tables:
        Available channels in tier

        {tier}_col_idx:
        file_db.columns[{tier}_col_idx] will return the list of columns available in tier's file
    """

    def __init__(self, config: str | dict, file_df: str = None, scan: bool = True):
        """
        Parameters
        ----------
        config
            Dict or path to JSON file specifying data directories, tiers, and file name templates

        file_df
            Path to LH5 file written by to_disk()

        scan
            True by default, whether the fileDB should scan the DAQ directory to
            fill its rows with file keys
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
            names = list(parse_arr[:, 1])  # fields required to generate file name
            names = [n for n in names if n]  # Remove none values
            names = list(np.unique(names))
            names += [f"{tier}_file" for tier in self.tiers]  # the generated file names
            names += [f"{tier}_size" for tier in self.tiers]  # file sizes
            names += ["file_status"]  # bonus columns

            self.df = pd.DataFrame(columns=names)

            self.columns = None

            if scan:
                self.scan_files()
                self.set_file_status()
                self.set_file_sizes()
        else:
            self.from_disk(config, file_df)

    def set_config(self, config: dict):
        """
        Helper function called by __init__()
        """
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

        for path, _folders, files in os.walk(scan_dir):
            n_files += len(files)

            for f in files:
                # in some cases, we need information from the path name
                if "/" in template:
                    f_tmp = path.replace(scan_dir, "") + "/" + f
                else:
                    f_tmp = f

                finfo = parse(template, f_tmp)
                if finfo is not None:
                    finfo = finfo.named
                    for tier in self.tiers:
                        finfo[f"{tier}_file"] = self.file_format[tier].format(**finfo)

                    file_keys.append(finfo)

        if n_files == 0:
            raise FileNotFoundError(f"No {low_tier} files found")

        if len(file_keys) == 0:
            raise FileNotFoundError(f"No {low_tier} files matched pattern ", template)

        temp_df = pd.DataFrame(file_keys)

        # fill the main DataFrame
        self.df = pd.concat([self.df, temp_df])

        # convert cols to numeric dtypes where possible
        for col in self.df.columns:
            self.df[col] = pd.to_numeric(self.df[col], errors="ignore")

    def set_file_status(self):
        """
        Add a column to the dataframe with a bit corresponding to whether each tier's file exists
        """

        def check_status(row):
            status = 0
            for i, tier in enumerate(self.tiers):
                path_name = (
                    self.data_dir + self.tier_dirs[tier] + "/" + row[f"{tier}_file"]
                )
                if os.path.exists(path_name):
                    status |= 1 << len(self.tiers) - i - 1

            return status

        self.df["file_status"] = self.df.apply(check_status, axis=1)

    def set_file_sizes(self):
        def get_size(row, tier):
            size = 0
            path_name = self.data_dir + self.tier_dirs[tier] + "/" + row[f"{tier}_file"]
            if os.path.exists(path_name):
                size = os.path.getsize(path_name)
            return size

        for tier in self.tiers:
            self.df[f"{tier}_size"] = self.df.apply(get_size, axis=1, tier=tier)

    def get_tables_columns(self, col_output: str = None):
        """
        Opens found files to save available tables and columns

        Adds the available tables in each tier as a column in fileDB
        by searching for group names that match the provided table_format
        and saving the associated keyword values

        Returns a table with each unique list of columns found in each table
        Adds a column to the FileDB dataframe df['{tier}_col_idx'] that maps to the column table

        Optionally write the column table to LH5 file as a VectorOfVectors
        """

        def update_tables_cols(row, tier):
            fpath = self.data_dir + self.tier_dirs[tier] + "/" + row[f"{tier}_file"]

            if os.path.exists(fpath):
                f = h5py.File(fpath)
            else:
                return pd.Series({f"{tier}_tables": None, f"{tier}_col_idx": None})

            # Get tables in each tier
            tier_tables = []
            template = self.table_format[tier]
            if template[-1] == "/":
                template = template[:-1]

            braces = list(re.finditer("{|}", template))

            if len(braces) > 2:
                raise ValueError("Tables can only have one identifier")
            if len(braces) % 2 != 0:
                raise ValueError("Braces mismatch in table format")
            if len(braces) == 0:
                tier_tables.append("")
            else:
                wildcard = (
                    template[: braces[0].span()[0]]
                    + "*"
                    + template[braces[1].span()[1] :]
                )

                groups = ls(f, wildcard)
                tier_tables = [
                    list(parse(template, g).named.values())[0] for g in groups
                ]

            # Get columns
            col_idx = []
            template = self.table_format[tier]
            fm = string.Formatter()

            for tb in tier_tables:
                parse_arr = np.array(list(fm.parse(template)))
                names = list(parse_arr[:, 1])
                if len(names) > 0:
                    keyword = names[0]
                    args = {keyword: tb}
                    table_name = template.format(**args)
                else:
                    table_name = template

                col = ls(f[table_name])
                if col not in columns:
                    columns.append(col)
                    col_idx.append(len(columns) - 1)
                else:
                    col_idx.append(columns.index(col))

            return pd.Series(
                {f"{tier}_tables": tier_tables, f"{tier}_col_idx": col_idx}
            )

        columns = []
        for tier in self.tiers:
            self.df[[f"{tier}_tables", f"{tier}_col_idx"]] = self.df.apply(
                update_tables_cols, axis=1, tier=tier
            )

        self.columns = columns

        if col_output is not None:
            flattened = []
            length = []
            for i, col in enumerate(columns):
                if i == 0:
                    length.append(len(col))
                else:
                    length.append(length[i - 1] + len(col))
                for c in col:
                    flattened.append(c)
            columns_vov = VectorOfVectors(
                flattened_data=flattened, cumulative_length=length
            )
            sto = LH5Store()
            sto.write_object(columns_vov, "unique_columns", col_output)

        return columns

    def from_disk(self, cfg_name: str, db_name: str):
        """
        Fills self.df and config with the information from a file created by to_lgdo()
        """
        with open(cfg_name) as cfg:
            config = json.load(cfg)
        self.set_config(config)
        self.df = pd.read_hdf(db_name, key="dataframe")
        sto = LH5Store()
        vov, _ = sto.read_object("columns", db_name)
        # Convert back from VoV of UTF-8 bytestrings to a list of lists of strings
        vov = list(vov)
        columns = []
        for ov in vov:
            columns.append([v.decode("utf-8") for v in ov])
        self.columns = columns

    def to_disk(self, cfg_name: str, db_name: str):
        """
        Writes config information to cfg_name and DataFrame to df_name

        Parameters
        -----------
        cfg_name
            Path to output JSON file for config

        db_name
            Path to output LH5 file for FileDB
        """
        with open(cfg_name, "w") as cfg:
            json.dump(self.config, cfg)

        if self.columns is not None:
            flat = []
            cum_l = [0]
            for i in range(len(self.columns)):
                flat += self.columns[i]
                cum_l.append(cum_l[i] + len(self.columns[i]))
            cum_l = cum_l[1:]
            # Must use type 'S' to play nice with HDF
            col_vov = VectorOfVectors(
                flattened_data=Array(nda=np.array(flat).astype("S")),
                cumulative_length=Array(nda=np.array(cum_l)),
            )
            sto = LH5Store()
            sto.write_object(col_vov, "columns", db_name, wo_mode="o")

        self.df.to_hdf(db_name, "dataframe")

    def scan_daq_files(self, verbose: bool = False):
        """
        Does the exact same thing as scan_files but with extra config arguments for a DAQ directory and template
        instead of using the lowest (raw) tier
        """
        file_keys = []
        n_files = 0

        for path, _folders, files in os.walk(self.daq_dir):
            n_files += len(files)

            for f in files:

                # in some cases, we need information from the path name
                if "/" in self.daq_template:
                    f_tmp = path.replace(self.daq_dir, "") + "/" + f
                else:
                    f_tmp = f

                finfo = parse(self.daq_template, f_tmp)
                if finfo is not None:
                    finfo = finfo.named
                    file_keys.append(finfo)
                for tier in self.tiers:
                    finfo[f"{tier}_file"] = self.file_format[tier].format(**finfo)

        if n_files == 0:
            raise FileNotFoundError("No daq files found")

        if len(file_keys) == 0:
            raise FileNotFoundError("No daq files matched pattern ", self.daq_template)

        temp_df = pd.DataFrame(file_keys)

        # fill the main DataFrame
        self.df = pd.concat([self.df, temp_df])

        # convert cols to numeric dtypes where possible
        for col in self.df.columns:
            self.df[col] = pd.to_numeric(self.df[col], errors="ignore")

        if verbose:
            log.info(self)

    def __repr__(self):
        string = "Columns: " + str(self.columns) + "\n"
        string += str(self.df)
        return string
