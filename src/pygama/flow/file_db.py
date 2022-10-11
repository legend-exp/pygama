"""
Utilities for LH5 file inventory.
"""
from __future__ import annotations

import json
import logging
import os
import re
import string
import warnings
from datetime import datetime, timezone

import h5py
import numpy as np
import pandas as pd
from parse import parse

from pygama.lgdo import Array, LH5Store, Scalar, VectorOfVectors
from pygama.lgdo.lh5_store import ls

log = logging.getLogger(__name__)


def to_datetime(key: str) -> datetime:
    """Convert LEGEND cycle key to :class:`~datetime.datetime`.

    Assumes `key` is formatted as ``YYYYMMDDTHHMMSSZ`` (UTC).
    """
    m = re.match(r"^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z$", key)
    if m is None:
        raise ValueError(f"Could not parse '{key}' as a datetime object")
    else:
        g = [int(el) for el in m.groups()]
        return datetime(*g, tzinfo=timezone.utc)


def to_unixtime(key: str) -> int:
    """Convert LEGEND cycle key to `POSIX timestamp <https://en.wikipedia.org/wiki/Unix_time>`_."""
    return int(to_datetime(key).timestamp())


class FileDB:
    """LH5 file database.

    A class containing a :class:`pandas.DataFrame` that has additional
    functions to scan the data directory, fill the dataframe's columns with
    information about each file, and read or write to disk in an LGDO format.

    The database contains the following columns:

    - file keys: the fields specified in the configuration file's
      ``file_format`` that are required to generate a file name e.g. ``run``,
      ``type``, ``timestamp`` etc.
    - ``{tier}_file``: generated file name for the tier.
    - ``{tier}_size``: size of file on disk, if applicable.
    - ``file_status``: contains a bit corresponding to whether or not a file
      for each tier exists for a given cycle e.g. If we have tiers `raw`,
      `dsp`, and `hit`, but only the `raw` file has been produced,
      ``file_status`` would be ``0b100``.
    - ``{tier}_tables``: available data streams (channels) in the tier.
    - ``{tier}_col_idx``: ``file_db.columns[{tier}_col_idx]`` will return the
      list of columns available in the tier's file.

    The database must be configured by a JSON file (or corresponding
    dictionary), which defines the data file names, paths and LH5 layout. For
    example:

    .. code-block:: json

        {
            "data_dir": "prod-ref-l200/generated/tier",
            "tier_dirs": {
                "raw": "/raw",
                "dsp": "/dsp",
                "hit": "/hit",
                "tcm": "/tcm",
                "evt": "/evt"
            },
            "file_format": {
                "raw": "/{type}/{period}/{run}/{exp}-{period}-{run}-{type}-{timestamp}-tier_raw.lh5",
                "dsp": "/{type}/{period}/{run}/{exp}-{period}-{run}-{type}-{timestamp}-tier_dsp.lh5",
                "hit": "/{type}/{period}/{run}/{exp}-{period}-{run}-{type}-{timestamp}-tier_hit.lh5",
                "evt": "/{type}/{period}/{run}/{exp}-{period}-{run}-{type}-{timestamp}-tier_evt.lh5",
                "tcm": "/{type}/{period}/{run}/{exp}-{period}-{run}-{type}-{timestamp}-tier_tcm.lh5"
            },
            "table_format": {
                "raw": "ch{ch:03d}/raw",
                "dsp": "ch{ch:03d}/dsp",
                "hit": "{ch}/hit",
                "evt": "{grp}/evt",
                "tcm": "hardware_tcm"
            },
            "tables": {
                "raw": [0, 1, 2, 4, 5, 6, 7],
                "dsp": [0, 1, 2, 4, 5, 6, 7],
                "hit": [0, 1, 2, 4, 5, 6, 7],
                "tcm": [""],
                "evt": [""]
            },
            "columns": {
                "raw": ["baseline", "waveform", "daqenergy"],
                "dsp": ["trapEftp", "AoE", "trapEmax"],
                "hit": ["trapEftp_cal", "trapEmax_cal"],
                "tcm": ["cumulative_length", "array_id", "array_idx"],
                "evt": ["lar_veto", "muon_veto", "ge_mult"]
            }
        }

    :class:`FileDB` objects can be also stored on disk and read-in at later
    times.

    Examples
    --------
    >>> from pygama.flow import FileDB
    >>> db = FileDB("./filedb_config.json")
    >>> db.get_tables_columns()  # read in also table columns names
    >>> print(db)
    << Columns >>
    [['baseline', 'card', 'ch_orca', 'channel', 'crate', 'daqenergy', 'deadtime', 'dr_maxticks', 'dr_start_pps', 'dr_start_ticks', 'dr_stop_pps', 'dr_stop_ticks', 'eventnumber', 'fcid', 'numtraces', 'packet_id', 'runtime', 'timestamp', 'to_abs_mu_usec', 'to_dt_mu_usec', 'to_master_sec', 'to_mu_sec', 'to_mu_usec', 'to_start_sec', 'to_start_usec', 'tracelist', 'ts_maxticks', 'ts_pps', 'ts_ticks', 'waveform'], ['bl_intercept', 'bl_mean', 'bl_slope', 'bl_std', 'tail_slope', 'tail_std', 'wf_blsub'], ['array_id', 'array_idx', 'cumulative_length']]
    << DataFrame >>
       exp period   run         timestamp type  ... hit_col_idx tcm_tables tcm_col_idx evt_tables evt_col_idx
    0  l60    p01  r014  20220716T105236Z  cal  ...        None         []         [2]       None        None
    1  l60    p01  r014  20220716T104550Z  cal  ...        None         []         [2]       None        None
    >>> db.to_disk("file_db.lh5")
    """

    def __init__(self, config: str | dict, scan: bool = True) -> None:
        """
        Parameters
        ----------
        config
            dictionary or path to JSON file specifying data directories, tiers,
            and file name templates. Can also be path to existing LH5 file
            containing :class:`FileDB` object serialized by :meth:`.to_disk()`.
        scan
            whether the file database should scan the directory containing
            `raw` files to fill its rows with file keys.
        """
        config_path = None
        if isinstance(config, str):
            if h5py.is_hdf5(config):
                self.from_disk(config)
                return
            else:
                config_path = config
                with open(config) as f:
                    config = json.load(f)

        if not isinstance(config, dict):
            raise ValueError("Bad FileDB configuration value")

        self.set_config(config, config_path)

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

            # Use config columns and tables if provided
            if "columns" in self.config.keys() and "tables" in self.config.keys():
                log.info("Setting columns/tables from config")
                self.columns = list(self.config["columns"].values())
                for tier in self.tiers:
                    self.df[f"{tier}_tables"] = [self.config["tables"][tier]] * len(
                        self.df
                    )
                    self.df[f"{tier}_col_idx"] = [
                        [self.columns.index(self.config["columns"][tier])]
                        * len(self.df[f"{tier}_tables"].iloc[0])
                    ] * len(self.df)

    def set_config(self, config: dict, config_path: str = None) -> None:
        """Read in the configuration dictionary."""
        self.config = config
        self.tiers = list(self.config["tier_dirs"].keys())
        self.file_format = self.config["file_format"]
        self.tier_dirs = self.config["tier_dirs"]
        self.table_format = self.config["table_format"]

        # Handle environment variables
        data_dir = os.path.expandvars(self.config["data_dir"])

        # Relative paths are interpreted relative to the configuration file
        if not data_dir.startswith("/"):
            config_dir = os.path.dirname(config_path)
            data_dir = os.path.join(config_dir, data_dir)
            data_dir = os.path.abspath(data_dir)
        self.data_dir = data_dir

    def scan_files(self) -> None:
        """Scan the directory containing files from the lower tier and fill the
        dataframe.

        The lower tier is defined as the first element of the `tiers` array.
        Only fills columns that can be populated with just the `raw` files.
        """
        file_keys = []
        n_files = 0
        low_tier = self.tiers[0]
        template = self.file_format[low_tier]
        scan_dir = self.data_dir + self.tier_dirs[low_tier]

        log.info(f"Scanning {scan_dir} with template {template}")

        for path, _folders, files in os.walk(scan_dir):
            log.debug(f"Scanning {path}")
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

        # sort rows according to timestamps
        log.debug("Sorting database entries according to timestamp")
        self.df["_datetime"] = self.df["timestamp"].apply(to_datetime)
        self.df.sort_values("_datetime", ignore_index=True, inplace=True)
        self.df.drop("_datetime", axis=1, inplace=True)

    def set_file_status(self) -> None:
        """
        Add a column to the dataframe with a bit corresponding to whether each
        tier's file exists.

        For example, if we have tiers `raw`, `dsp`, and `hit`, but only the
        `raw` file has been produced, ``file_status`` would be 4 (``0b100`` in
        binary representation).
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

    def set_file_sizes(self) -> None:
        """
        Add columns (for each tier) to the database containing the
        corresponding file size in bytes as reported by
        :func:`os.path.getsize`.
        """

        def get_size(row, tier):
            size = 0
            path_name = self.data_dir + self.tier_dirs[tier] + "/" + row[f"{tier}_file"]
            if os.path.exists(path_name):
                size = os.path.getsize(path_name)
            return size

        for tier in self.tiers:
            self.df[f"{tier}_size"] = self.df.apply(get_size, axis=1, tier=tier)

    def scan_tables_columns(
        self, to_file: str = None, override: bool = False
    ) -> list[str]:
        """Open files in the database to read (and store) available tables (and
        columns therein) names.

        Adds the available table names in each tier as a column in the
        dataframe by searching for group names that match the configured
        ``table_format`` and saving the associated keyword values.

        Returns a list with each unique list of columns found in each table
        and adds a column ``{tier}_col_idx`` to the dataframe that maps to the
        column table.

        Parameters
        ----------
        to_file:
            Optionally write the column table to an LH5 file (as a
            :class:`~.lgdo.vectorofvectors.VectorOfVectors`)

        override:
            If the FileDB already has a `columns` field, the scan will not run unless
            this parameter is set to True
        """
        log.info("Getting table column names")

        if self.columns is not None:
            if not override:
                log.warning(
                    "LH5 tables/columns names already set, if you want to perform the scan anyway, set override=True"
                )
                return
            else:
                log.warning("Overwriting existing LH5 tables/columns names")

        def update_tables_cols(row, tier: str) -> pd.Series:
            fpath = self.data_dir + self.tier_dirs[tier] + "/" + row[f"{tier}_file"]

            log.debug(f"Reading column names for tier '{tier}' from {fpath}")

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

                # TODO this call here is really expensive!
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

        if to_file is not None:
            log.debug(f"Writing column names to '{to_file}'")
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
            sto.write_object(columns_vov, "unique_columns", to_file)

        return columns

    def from_disk(self, filename: str) -> None:
        """
        Fills the dataframe (and configuration dictionary) with the information
        from a file created by :meth:`to_disk`.
        """
        sto = LH5Store()
        cfg, _ = sto.read_object("config", filename)
        self.set_config(json.loads(cfg.value.decode()))

        self.df = pd.read_hdf(filename, key="dataframe")

        vov, _ = sto.read_object("columns", filename)
        # Convert back from VoV of UTF-8 bytestrings to a list of lists of strings
        vov = list(vov)
        columns = []
        for ov in vov:
            columns.append([v.decode("utf-8") for v in ov])
        self.columns = columns

    def to_disk(self, filename: str, wo_mode="write_safe") -> None:
        """Serializes database to disk.

        Parameters
        -----------
        filename
            output LH5 file name.
        wo_mode
            passed to :meth:`~.lgdo.lh5_store.write_object`.
        """
        log.debug(f"Writing database to {filename}")

        sto = LH5Store()
        sto.write_object(
            Scalar(json.dumps(self.config)), "config", filename, wo_mode=wo_mode
        )

        if wo_mode in ["write_safe", "w", "overwrite_file", "of"]:
            wo_mode = "a"

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
            sto.write_object(col_vov, "columns", filename, wo_mode=wo_mode)

        # FIXME: to_hdf() throws this:
        #
        #     pandas.errors.PerformanceWarning: your performance may suffer as
        #     PyTables will pickle object types that it cannot map directly to c-types
        #
        # not sure how to fix this so we ignore the warning for the moment
        warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
        self.df.to_hdf(filename, key="dataframe", format="fixed", mode="r+")

    def scan_daq_files(self, daq_dir: str, daq_template: str) -> None:
        """
        Does the exact same thing as :meth:`.scan_files` but with extra
        configuration arguments for a DAQ directory and template instead of
        using the lowest tier.
        """
        file_keys = []
        n_files = 0

        for path, _folders, files in os.walk(daq_dir):
            n_files += len(files)

            for f in files:

                # in some cases, we need information from the path name
                if "/" in daq_template:
                    f_tmp = path.replace(daq_dir, "") + "/" + f
                else:
                    f_tmp = f

                finfo = parse(daq_template, f_tmp)
                if finfo is not None:
                    finfo = finfo.named
                    file_keys.append(finfo)
                for tier in self.tiers:
                    finfo[f"{tier}_file"] = self.file_format[tier].format(**finfo)

        if n_files == 0:
            raise FileNotFoundError("No DAQ files found")

        if len(file_keys) == 0:
            raise FileNotFoundError("No DAQ files matched pattern ", daq_template)

        temp_df = pd.DataFrame(file_keys)

        # fill the main DataFrame
        self.df = pd.concat([self.df, temp_df])

        # convert cols to numeric dtypes where possible
        for col in self.df.columns:
            self.df[col] = pd.to_numeric(self.df[col], errors="ignore")

    def __repr__(self) -> str:
        string = (
            "<< Columns >>\n" + self.columns.__repr__() + "\n"
            "\n"
            "<< DataFrame >>\n" + self.df.__repr__()
        )
        return string
