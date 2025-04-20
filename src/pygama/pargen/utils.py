from __future__ import annotations

import logging
from types import FunctionType

import numpy as np
import pandas as pd
from iminuit import Minuit, cost
from lgdo import lh5

log = logging.getLogger(__name__)


def convert_to_minuit(pars, func):
    try:
        c = cost.UnbinnedNLL(np.array([0]), func.pdf_ext)
    except AttributeError:
        c = cost.UnbinnedNLL(np.array([0]), func)
    if isinstance(pars, dict):
        m = Minuit(c, **pars)
    else:
        m = Minuit(c, *pars)
    return m


def return_nans(input):
    if isinstance(input, FunctionType):
        args = input.__code__.co_varnames[: input.__code__.co_argcount][1:]
        m = convert_to_minuit(np.full(len(args), np.nan), input)
        return m.values, m.errors, np.full((len(m.values), len(m.values)), np.nan)
    else:
        args = input.required_args()
        m = convert_to_minuit(np.full(len(args), np.nan), input)
        return m.values, m.errors, np.full((len(m.values), len(m.values)), np.nan)


def load_data(
    files: str | list | dict,
    lh5_path: str,
    cal_dict: dict,
    params: set,
    cal_energy_param: str = "cuspEmax_ctc_cal",
    threshold=None,
    return_selection_mask=False,
) -> pd.DataFrame | tuple(pd.DataFrame, np.array):
    """
    Loads parameters from data files. Applies calibration to cal_energy_param
    and uses this to apply a lower energy threshold.

    files
        file or list of files or dict pointing from timestamps to lists of files
    lh5_path
        path to table in files
    cal_dict
        dictionary with operations used to apply calibration constants
    params
        list of parameters to load from file
    cal_energy_param
        name of uncalibrated energy parameter
    threshold
        lower energy threshold for events to load
    return_selection_map
        if True, return selection mask for threshold along with data
    """

    params = set(params)
    if isinstance(files, str):
        files = [files]

    if isinstance(files, dict):
        # Go through each tstamp and recursively load_data on file lists
        df = []
        masks = []
        for tstamp, tfiles in files.items():
            file_df = load_data(
                tfiles,
                lh5_path,
                cal_dict.get(tstamp, cal_dict),
                params,
                cal_energy_param,
                threshold,
                return_selection_mask,
            )

            if return_selection_mask:
                file_df[0]["run_timestamp"] = np.full(
                    len(file_df[0]), tstamp, dtype=object
                )
                df.append(file_df[0])
                masks.append(file_df[1])
            else:
                file_df["run_timestamp"] = np.full(len(file_df), tstamp, dtype=object)
                df.append(file_df)

        df = pd.concat(df)
        if return_selection_mask:
            masks = np.concatenate(masks)

    elif isinstance(files, list):
        # Get set of available fields between input table and cal_dict
        file_keys = lh5.ls(
            files[0], lh5_path if lh5_path[-1] == "/" else lh5_path + "/"
        )
        file_keys = {key.split("/")[-1] for key in file_keys}

        # Get set of keys in calibration expressions that show up in file
        cal_keys = {
            name
            for info in cal_dict.values()
            for name in compile(info["expression"], "0vbb is real!", "eval").co_names
        } & file_keys

        # Get set of fields to read from files
        fields = cal_keys | (file_keys & params)

        lh5_it = lh5.iterator.LH5Iterator(
            files, lh5_path, field_mask=fields, buffer_len=100000
        )
        df_fields = params & (fields | set(cal_dict))
        if df_fields != params:
            log.debug(
                f"load_data(): params not found in data files or cal_dict: {params-df_fields}"
            )
        df = pd.DataFrame(columns=list(df_fields))

        for table in lh5_it:
            # Evaluate all provided expressions and add to table
            for outname, info in cal_dict.items():
                table[outname] = table.eval(
                    info["expression"], info.get("parameters", None)
                )
            entry = lh5_it.current_global_entries[0]
            n_rows = len(table)

            # Copy params in table into dataframe
            for par in df:
                # First set of entries: allocate enough memory for all entries
                if entry == 0:
                    df[par] = np.resize(table[par], len(lh5_it))
                else:
                    df.loc[entry : entry + n_rows - 1, par] = table[par][:n_rows]

        # Evaluate threshold mask and drop events below threshold
        if threshold is not None:
            masks = df[cal_energy_param] > threshold
            df.drop(np.where(~masks)[0], inplace=True)
        else:
            masks = np.ones(len(df), dtype=bool)

    log.debug("data loaded")
    if return_selection_mask:
        return df, masks
    else:
        return df
