from __future__ import annotations

import logging
from types import FunctionType

import lgdo.lh5_store as lh5
import numpy as np
import pandas as pd
from iminuit import Minuit, cost, util

import pygama.pargen.cuts as cts

log = logging.getLogger(__name__)


def return_nans(input):
    if isinstance(input, FunctionType):
        args = input.__code__.co_varnames[: input.__code__.co_argcount][1:]
        c = cost.UnbinnedNLL(np.array([0]), input)
        m = Minuit(c, *[np.nan for arg in args])
        return m.values, m.errors, np.full((len(m.values), len(m.values)), np.nan)
    else:
        args = input.pdf.__code__.co_varnames[: input.pdf.__code__.co_argcount][1:]
        c = cost.UnbinnedNLL(np.array([0]), input.pdf)
        m = Minuit(c, *[np.nan for arg in args])
        return m.values, m.errors, np.full((len(m.values), len(m.values)), np.nan)


def tag_pulser(files, lh5_path):
    pulser_df = lh5.load_dfs(files, ["timestamp", "trapTmax"], lh5_path)
    pulser_props = cts.find_pulser_properties(pulser_df, energy="trapTmax")
    if len(pulser_props) > 0:
        final_mask = None
        for entry in pulser_props:
            e_cut = (pulser_df.trapTmax.values < entry[0] + entry[1]) & (
                pulser_df.trapTmax.values > entry[0] - entry[1]
            )
            if final_mask is None:
                final_mask = e_cut
            else:
                final_mask = final_mask | e_cut
        ids = ~(final_mask)
        log.debug(f"pulser found: {pulser_props}")
    else:
        ids = np.ones(len(pulser_df), dtype=bool)
        log.debug(f"no pulser found")
    return ids


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
    params=["cuspEmax"],
    cal_energy_param: str = "cuspEmax_ctc_cal",
    threshold=None,
) -> tuple(np.array, np.array, np.array, np.array):
    """
    Loads in the A/E parameters needed and applies calibration constants to energy
    """

    sto = lh5.LH5Store()

    if isinstance(files, dict):
        df = []
        all_files = []
        masks = np.array([], dtype=bool)
        for tstamp, tfiles in files.items():
            table = sto.read_object(lh5_path, tfiles)[0]
            if tstamp in cal_dict:
                file_df = table.eval(cal_dict[tstamp]).get_dataframe()
            else:
                file_df = table.eval(cal_dict).get_dataframe()
            file_df["timestamp"] = np.full(len(file_df), tstamp, dtype=object)
            params.append("timestamp")
            if threshold is not None:
                mask = file_df[cal_energy_param] < threshold

                file_df.drop(np.where(mask)[0], inplace=True)
            else:
                mask = np.zeros(len(file_df), dtype=bool)
            masks = np.append(masks, ~mask)
            df.append(file_df)
            all_files += tfiles

        df = pd.concat(df)

    elif isinstance(files, list):
        table = sto.read_object(lh5_path, files)[0]
        df = table.eval(cal_dict).get_dataframe()
        if threshold is not None:
            masks = df[cal_energy_param] > threshold
            df.drop(np.where(~masks)[0], inplace=True)
        else:
            masks = np.ones(len(df), dtype=bool)
        all_files = files

    if lh5_path[-1] != "/":
        lh5_path += "/"
    keys = lh5.ls(all_files[0], lh5_path)
    keys = [key.split("/")[-1] for key in keys]
    params = get_params(keys + list(df.keys()), params)

    ids = tag_pulser(all_files, lh5_path)
    df["is_not_pulser"] = ids[masks]
    params.append("is_not_pulser")

    for col in list(df.keys()):
        if col not in params:
            df.drop(col, inplace=True, axis=1)

    param_dict = {}
    for param in params:
        if param not in df:
            df[param] = lh5.load_nda(all_files, [param], lh5_path)[param][masks]
    log.debug(f"data loaded")
    return df
