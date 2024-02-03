from __future__ import annotations

import logging
from types import FunctionType

import numpy as np
import pandas as pd
from iminuit import Minuit, cost, util
from lgdo import Table, lh5

log = logging.getLogger(__name__)
sto = lh5.LH5Store()


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

    out_df = pd.DataFrame(columns=params)

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
        data = pd.DataFrame(
            {
                "array_id": sto.read("hardware_tcm_1/array_id", tcm_file)[0].view_as(
                    "np"
                ),
                "array_idx": sto.read("hardware_tcm_1/array_idx", tcm_file)[0].view_as(
                    "np"
                ),
            }
        )
        cumulength = sto.read("hardware_tcm_1/cumulative_length", tcm_file)[0].view_as(
            "np"
        )
        cumulength = np.append(np.array([0]), cumulength)
        n_channels = np.diff(cumulength)
        evt_numbers = np.repeat(np.arange(0, len(cumulength) - 1), np.diff(cumulength))
        evt_mult = np.repeat(np.diff(cumulength), np.diff(cumulength))
        data["evt_number"] = evt_numbers
        data["evt_mult"] = evt_mult
        high_mult_events = np.where(n_channels > multiplicity_threshold)[0]

        ids = data.query(f"array_id=={channel} and evt_number in @high_mult_events")[
            "array_idx"
        ].to_numpy()
        mask = np.zeros(len(data.query(f"array_id=={channel}")), dtype="bool")
        mask[ids] = True
    return ids, mask
