from __future__ import annotations

import logging
from types import FunctionType

import lgdo.lh5_store as lh5
import numpy as np
import pandas as pd
from iminuit import Minuit, cost, util

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
    return_selection_mask=False,
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
            file_df["run_timestamp"] = np.full(len(file_df), tstamp, dtype=object)
            params.append("run_timestamp")
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

    for col in list(df.keys()):
        if col not in params:
            df.drop(col, inplace=True, axis=1)

    param_dict = {}
    for param in params:
        if param not in df:
            df[param] = lh5.load_nda(all_files, [param], lh5_path)[param][masks]
    log.debug(f"data loaded")
    if return_selection_mask:
        return df, masks
    else:
        return df


def get_tcm_pulser_ids(tcm_file, channel, multiplicity_threshold):
    if isinstance(channel, str):
        if channel[:2] == "ch":
            chan = int(channel[2:])
        else:
            chan = int(channel)
    else:
        chan = channel
    if isinstance(tcm_file, list):
        mask = np.array([], dtype=bool)
        for file in tcm_file:
            _, file_mask = get_tcm_pulser_ids(file, chan, multiplicity_threshold)
            mask = np.append(mask, file_mask)
        ids = np.where(mask)[0]
    else:
        data = lh5.load_dfs(tcm_file, ["array_id", "array_idx"], "hardware_tcm_1")
        cum_length = lh5.load_nda(tcm_file, ["cumulative_length"], "hardware_tcm_1")[
            "cumulative_length"
        ]
        cum_length = np.append(np.array([0]), cum_length)
        n_channels = np.diff(cum_length)
        evt_numbers = np.repeat(np.arange(0, len(cum_length) - 1), np.diff(cum_length))
        evt_mult = np.repeat(np.diff(cum_length), np.diff(cum_length))
        data["evt_number"] = evt_numbers
        data["evt_mult"] = evt_mult
        high_mult_events = np.where(n_channels > multiplicity_threshold)[0]

        ids = data.query(f"array_id=={channel} and evt_number in @high_mult_events")[
            "array_idx"
        ].to_numpy()
        mask = np.zeros(len(data.query(f"array_id=={channel}")), dtype="bool")
        mask[ids] = True
    return ids, mask
