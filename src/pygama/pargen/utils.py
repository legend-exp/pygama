from __future__ import annotations

import logging
from types import FunctionType

import numpy as np
import pandas as pd
from iminuit import Minuit, cost
from lgdo import lh5

log = logging.getLogger(__name__)
sto = lh5.LH5Store()


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


def get_params(file_params, param_list):
    out_params = []
    if isinstance(file_params, dict):
        possible_keys = file_params.keys()
    elif isinstance(file_params, list):
        possible_keys = file_params
    for param in param_list:
        for key in possible_keys:
            if key in param:
                out_params.append(key)
    return np.unique(out_params).tolist()


def load_data(
    files: list,
    lh5_path: str,
    cal_dict: dict,
    params: list,
    cal_energy_param: str = "cuspEmax_ctc_cal",
    threshold=None,
    return_selection_mask=False,
) -> tuple(np.array, np.array, np.array, np.array):
    """
    Loads in the A/E parameters needed and applies calibration constants to energy
    """

    if isinstance(files, str):
        files = [files]

    if isinstance(files, dict):
        keys = lh5.ls(
            files[list(files)[0]][0],
            lh5_path if lh5_path[-1] == "/" else lh5_path + "/",
        )
        keys = [key.split("/")[-1] for key in keys]
        if list(files)[0] in cal_dict:
            params = get_params(keys + list(cal_dict[list(files)[0]].keys()), params)
        else:
            params = get_params(keys + list(cal_dict.keys()), params)

        df = []
        all_files = []
        masks = np.array([], dtype=bool)
        for tstamp, tfiles in files.items():
            table = sto.read(lh5_path, tfiles)[0]

            file_df = pd.DataFrame(columns=params)
            if tstamp in cal_dict:
                cal_dict_ts = cal_dict[tstamp]
            else:
                cal_dict_ts = cal_dict

            for outname, info in cal_dict_ts.items():
                outcol = table.eval(info["expression"], info.get("parameters", None))
                table.add_column(outname, outcol)

            for param in params:
                file_df[param] = table[param]

            file_df["run_timestamp"] = np.full(len(file_df), tstamp, dtype=object)

            if threshold is not None:
                mask = file_df[cal_energy_param] > threshold
                file_df.drop(np.where(~mask)[0], inplace=True)
            else:
                mask = np.ones(len(file_df), dtype=bool)
            masks = np.append(masks, mask)
            df.append(file_df)
            all_files += tfiles

        params.append("run_timestamp")
        df = pd.concat(df)

    elif isinstance(files, list):
        keys = lh5.ls(files[0], lh5_path if lh5_path[-1] == "/" else lh5_path + "/")
        keys = [key.split("/")[-1] for key in keys]
        params = get_params(keys + list(cal_dict.keys()), params)

        table = sto.read(lh5_path, files)[0]
        df = pd.DataFrame(columns=params)
        for outname, info in cal_dict.items():
            outcol = table.eval(info["expression"], info.get("parameters", None))
            table.add_column(outname, outcol)
        for param in params:
            df[param] = table[param]
        if threshold is not None:
            masks = df[cal_energy_param] > threshold
            df.drop(np.where(~masks)[0], inplace=True)
        else:
            masks = np.ones(len(df), dtype=bool)
        all_files = files

    for col in list(df.keys()):
        if col not in params:
            df.drop(col, inplace=True, axis=1)

    log.debug("data loaded")
    if return_selection_mask:
        return df, masks
    else:
        return df
