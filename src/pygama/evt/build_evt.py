"""
This module implements routines to build the evt tier.
"""
from __future__ import annotations

import itertools
import json
import logging
import os
import re
from importlib import import_module

import lgdo.lh5_store as store
import numpy as np
from lgdo import Array, VectorOfVectors

log = logging.getLogger(__name__)


def num_and_pars(value: str, par_dic: dict):
    # function tries to convert a string to a int, float, bool
    # or returns the value if value is a key in par_dic
    if value in par_dic.keys():
        return par_dic[value]
    try:
        value = int(value)
    except ValueError:
        try:
            value = float(value)
        except ValueError:
            try:
                value = bool(value)
            except ValueError:
                pass
    return value


def evaluate_expression(
    f_tcm: str,
    f_evt: str,
    f_hit: str,
    f_dsp: str,
    chns: list,
    mod: str | list,
    expr: str,
    nrows: int,
    group: str,
    para: dict = None,
    qry: str = None,
    defv=np.nan,
) -> dict:
    """
    Evaluates the expression defined by the user across all channels according to the mode

    Parameters
    ----------
    f_tcm
       Path to tcm tier file
    f_evt
       Path to event tier file
    f_hit
       Path to hit tier file
    f_dsp
       Path to dsp tier file
    chns
       List of channel names across which expression gets evaluated (form: "ch<rawid>")
    mode
       The mode determines how the event entry is calculated across channels. Options are:
       - "first": The value of the channel in an event triggering first in time (according to tp_0_est) is returned. It  is possible to add a condition (e.g. "first>10"). Only channels fulfilling this condition are considered in the time evaluation. If no channel fullfilles the condition, nan is returned for this event.
       - "last": The value of the channel in an event triggering last in time (according to tp_0_est) is returned. It  is possible to add a condition (e.g. "last>10"). Only channels fulfilling this condition are considered in the time evaluation. If no channel fullfilles the condition, nan is returned for this event.
       - "tot": The sum of all channels across an event. It  is possible to add a condition (e.g. "tot>10"). Only channels fulfilling this condition are considered in the time evaluation. If no channel fullfilles the condition, zero is returned for this event. Booleans are treated as integers 0/1.
       - "any": Logical or between all channels. Non boolean values are True for values != 0 and False for values == 0.
       - "all": Logical and between all channels. Non boolean values are True for values != 0 and False for values == 0.
       - ch_field: A previously generated channel_id field (i.e. from the get_ch flag) can be given here, and the value of this specific channels is used. if ch_field is a VectorOfVectors, the channel list is ignored. If ch_field is an Array, the intersection of the passed channels list and the Array is formed. If a channel is not in the Array, the default is used.
       - "vov": Channels are not combined, but result saved as VectorOfVectors. Use of getch is recommended. It  is possible (and recommended) to add a condition (e.g. "vov>10"). Only channels fulfilling this condition are saved.
    qry
       A query that can set a condition on mode. Can be any tier (i.e. a channelxevents shaped boolean matrix for tiers below event or an events long boolean array at the evt level)
    expr
       The expression. That can be any mathematical equation/comparison. If mode == func, the expression needs to be a special processing function defined in modules (e.g. "modules.spm.get_energy). In the expression parameters from either hit, dsp, evt tier (from operations performed before this one! --> JSON operations order matters), or from the "parameters" field can be used.
    nrows
       Number of rows to be processed.
    group
       lh5 root group name
    dsp_group
        lh5 root group in dsp file
    hit_group
        lh5 root group in hit file
    para
       Dictionary of parameters defined in the "parameters" field in the configuration JSON file.
    defv
       default value of evaluation
    """

    # set modus variables
    mode, sorter = mod, None
    if isinstance(mod, list):
        mode = mod[0]
        sorter = mod[1].split(".")

    # find parameters in evt file or in parameters
    exprl = re.findall(r"(evt|hit|dsp).([a-zA-Z_$][\w$]*)", expr)
    var_ph = {}
    if os.path.exists(f_evt):
        var_ph = load_vars_to_nda(f_evt, "", exprl)
    if para:
        var_ph = var_ph | para

    if mode == "function":
        # evaluate expression
        func, params = expr.split("(")
        params = (
            params.replace("dsp.", "dsp_")
            .replace("hit.", "hit_")
            .replace("evt.", "evt_")
        )
        params = [f_hit, f_dsp, f_tcm, chns] + [
            num_and_pars(e, var_ph) for e in params[:-1].split(",")
        ]

        # load function dynamically
        p, m = func.rsplit(".", 1)
        met = getattr(import_module(p, package=__package__), m)
        out = met(*params)
        return {"values": out}

    else:
        # check if query is either on channel basis or evt basis (and not a mix)
        qry_mask = qry
        if qry is not None:
            if "evt." in qry and ("hit." in qry or "dsp." in qry):
                raise ValueError("Query can't be a mix of evt tier and lower tiers.")

            # if it is an evt query we can evaluate it directly here
            if os.path.exists(f_evt) and "evt." in qry:
                var_qry = load_vars_to_nda(
                    f_evt, "", re.findall(r"(evt).([a-zA-Z_$][\w$]*)", qry)
                )
                qry_mask = eval(qry.replace("evt.", "evt_"), var_qry)

        # load TCM data to define an event
        nda = store.load_nda(f_tcm, ["array_id", "array_idx"], "hardware_tcm_1/")
        ids = nda["array_id"]
        idx = nda["array_idx"]

        # switch through modes
        if (
            os.path.exists(f_evt)
            and "evt." == mode[:4]
            and mode.split(".")[-1]
            in [e.split("/")[-1] for e in store.ls(f_evt, "/evt/")]
        ):
            lstore = store.LH5Store()
            ch_comp, _ = lstore.read_object(mode.replace(".", "/"), f_evt)
            if isinstance(ch_comp, Array):
                return evaluate_at_channel(
                    idx,
                    ids,
                    f_hit,
                    f_dsp,
                    chns,
                    expr,
                    exprl,
                    ch_comp,
                    var_ph,
                    defv,
                )
            elif isinstance(ch_comp, VectorOfVectors):
                return evaluate_at_channel_vov(
                    idx,
                    ids,
                    f_hit,
                    f_dsp,
                    expr,
                    exprl,
                    ch_comp,
                    var_ph,
                )
            else:
                raise NotImplementedError(
                    type(ch_comp)
                    + " not supported (only Array and VectorOfVectors are supported)"
                )

        elif "first" == mode:
            return evaluate_to_first(
                idx,
                ids,
                f_hit,
                f_dsp,
                chns,
                expr,
                exprl,
                qry_mask,
                nrows,
                sorter,
                var_ph,
                defv,
            )
        elif "last" == mode:
            return evaluate_to_last(
                idx,
                ids,
                f_hit,
                f_dsp,
                chns,
                expr,
                exprl,
                qry_mask,
                nrows,
                sorter,
                var_ph,
                defv,
            )
        elif "sum" == mode:
            return evaluate_to_tot(
                idx,
                ids,
                f_hit,
                f_dsp,
                chns,
                expr,
                exprl,
                qry_mask,
                nrows,
                var_ph,
                defv,
            )
        elif "vov" == mode:
            return evaluate_to_vector(
                idx,
                ids,
                f_hit,
                f_dsp,
                chns,
                expr,
                exprl,
                qry_mask,
                nrows,
                var_ph,
            )
        elif "any" == mode:
            return evaluate_to_any(
                idx,
                ids,
                f_hit,
                f_dsp,
                chns,
                expr,
                exprl,
                qry_mask,
                nrows,
                var_ph,
                defv,
            )
        elif "all" == mode:
            return evaluate_to_all(
                idx,
                ids,
                f_hit,
                f_dsp,
                chns,
                expr,
                exprl,
                qry_mask,
                nrows,
                var_ph,
                defv,
            )
        else:
            raise ValueError(mode + " not a valid mode")


def find_parameters(
    f_hit: str,
    f_dsp: str,
    ch: str,
    idx_ch: np.ndarray,
    exprl: list,
) -> dict:
    # find fields in either dsp, hit
    var = load_vars_to_nda(f_hit, ch, exprl, idx_ch)
    dsp_dic = load_vars_to_nda(f_dsp, ch, exprl, idx_ch)

    return dsp_dic | var


def load_vars_to_nda(
    f_evt: str, group: str, exprl: list, idx: np.ndarray = None
) -> dict:
    lstore = store.LH5Store()
    var = {
        f"{e[0]}_{e[1]}": lstore.read_object(
            f"{group.replace('/','')}/{e[0]}/{e[1]}",
            f_evt,
            idx=idx,
        )[0]
        for e in exprl
        if e[1]
        in [
            x.split("/")[-1]
            for x in store.ls(f_evt, f"{group.replace('/','')}/{e[0]}/")
        ]
    }

    # to make any operations to VoVs we have to blow it up to a table (future change to more intelligant way)
    arr_keys = []
    for key, value in var.items():
        if isinstance(value, VectorOfVectors):
            var[key] = value.to_aoesa().nda
        elif isinstance(value, Array):
            var[key] = value.nda
            if var[key].ndim > 2:
                raise ValueError("Dim > 2 not supported")
            if var[key].ndim == 1:
                arr_keys.append(key)
        else:
            raise ValueError(f"{type(value)} not supported")

    # now we also need to set dimensions if we have an expression
    # consisting of a mix of VoV and Arrays
    if len(arr_keys) > 0 and not set(arr_keys) == set(var.keys()):
        for key in arr_keys:
            var[key] = var[key][:, None]

    log.debug(f"Found parameters {var.keys()}")
    return var


def evaluate_to_first(
    idx: np.ndarray,
    ids: np.ndarray,
    f_hit: str,
    f_dsp: str,
    chns: list,
    expr: str,
    exprl: list,
    qry: str | np.ndarray,
    nrows: int,
    sorter: list,
    var_ph: dict = None,
    defv=np.nan,
) -> dict:
    # define dimension of output array
    out = np.full(nrows, defv, dtype=type(defv))
    out_chs = np.zeros(len(out), dtype=int)
    outt = np.zeros(len(out))

    for ch in chns:
        # get index list for this channel to be loaded
        idx_ch = idx[ids == int(ch[2:])]

        if "tcm.array_id" == expr:
            res = np.full(len(out), int(ch[2:]), dtype=int)
        else:
            var = find_parameters(f_hit, f_dsp, ch, idx_ch, exprl) | var_ph

            # evaluate expression
            # move tier+dots in expression to underscores (e.g. evt.foo -> evt_foo)
            res = eval(
                expr.replace("dsp.", "dsp_")
                .replace("hit.", "hit_")
                .replace("evt.", "evt_"),
                var,
            )

            # if it is not a nparray it could be a single value
            # expand accordingly
            if not isinstance(res, np.ndarray):
                res = np.full(len(out), res, dtype=type(res))

        # get sub evt based query condition if needed
        if isinstance(qry, str):
            qry_lst = re.findall(r"(hit|dsp).([a-zA-Z_$][\w$]*)", qry)
            qry_var = find_parameters(f_hit, f_dsp, ch, idx_ch, qry_lst)
            limarr = eval(qry.replace("dsp.", "dsp_").replace("hit.", "hit_"), qry_var)

        # or forward the array
        elif isinstance(qry, np.ndarray):
            limarr = qry

        # if no condition, it must be true
        else:
            limarr = np.ones(len(res)).astype(bool)

        if limarr.dtype != bool:
            limarr = limarr.astype(bool)

        # append to out according to mode == first
        if ch == chns[0]:
            outt[:] = np.inf

        # find if sorter is in hit or dsp
        t0 = store.load_nda(
            f_hit if "hit" == sorter[0] else f_dsp,
            [sorter[1]],
            f"{ch}/{sorter[0]}/",
            idx_ch,
        )[sorter[1]]

        out[idx_ch] = np.where((t0 < outt) & (limarr), res, out[idx_ch])
        out_chs[idx_ch] = np.where((t0 < outt) & (limarr), int(ch[2:]), out_chs[idx_ch])
        outt[idx_ch] = np.where((t0 < outt) & (limarr), t0, outt[idx_ch])

    return {"values": out, "channels": out_chs}


def evaluate_to_last(
    idx: np.ndarray,
    ids: np.ndarray,
    f_hit: str,
    f_dsp: str,
    chns: list,
    expr: str,
    exprl: list,
    qry: str | np.ndarray,
    nrows: int,
    sorter: list,
    var_ph: dict = None,
    defv=np.nan,
) -> dict:
    # define dimension of output array
    out = np.full(nrows, defv, dtype=type(defv))
    out_chs = np.zeros(len(out), dtype=int)
    outt = np.zeros(len(out))

    for ch in chns:
        # get index list for this channel to be loaded
        idx_ch = idx[ids == int(ch[2:])]
        if "tcm.array_id" == expr:
            res = np.full(len(out), int(ch[2:]), dtype=int)
        else:
            # find fields in either dsp, hit
            var = find_parameters(f_hit, f_dsp, ch, idx_ch, exprl) | var_ph

            # evaluate expression
            # move tier+dots in expression to underscores (e.g. evt.foo -> evt_foo)
            res = eval(
                expr.replace("dsp.", "dsp_")
                .replace("hit.", "hit_")
                .replace("evt.", "evt_"),
                var,
            )

            # if it is not a nparray it could be a single value
            # expand accordingly
            if not isinstance(res, np.ndarray):
                res = np.full(len(out), res, dtype=type(res))

        # get sub evt based query condition if needed
        if isinstance(qry, str):
            qry_lst = re.findall(r"(hit|dsp).([a-zA-Z_$][\w$]*)", qry)
            qry_var = find_parameters(f_hit, f_dsp, ch, idx_ch, qry_lst)
            limarr = eval(qry.replace("dsp.", "dsp_").replace("hit.", "hit_"), qry_var)

        # or forward the array
        elif isinstance(qry, np.ndarray):
            limarr = qry

        # if no condition, it must be true
        else:
            limarr = np.ones(len(res)).astype(bool)
        if limarr.dtype != bool:
            limarr = limarr.astype(bool)
        # append to out according to mode == last
        # find if sorter is in hit or dsp
        t0 = store.load_nda(
            f_hit if "hit" == sorter[0] else f_dsp,
            [sorter[1]],
            f"{ch}/{sorter[0]}/",
            idx_ch,
        )[sorter[1]]

        out[idx_ch] = np.where((t0 > outt) & (limarr), res, out[idx_ch])
        out_chs[idx_ch] = np.where((t0 > outt) & (limarr), int(ch[2:]), out_chs[idx_ch])
        outt[idx_ch] = np.where((t0 > outt) & (limarr), t0, outt[idx_ch])

    return {"values": out, "channels": out_chs}


def evaluate_to_tot(
    idx: np.ndarray,
    ids: np.ndarray,
    f_hit: str,
    f_dsp: str,
    chns: list,
    expr: str,
    exprl: list,
    qry: str | np.ndarray,
    nrows: int,
    var_ph: dict = None,
    defv=np.nan,
) -> dict:
    # define dimension of output array
    out = np.full(nrows, defv, dtype=type(defv))

    for ch in chns:
        # get index list for this channel to be loaded
        idx_ch = idx[ids == int(ch[2:])]

        if "tcm.array_id" == expr:
            res = np.full(len(out), int(ch[2:]), dtype=int)
        else:
            # find fields in either dsp, hit
            var = find_parameters(f_hit, f_dsp, ch, idx_ch, exprl) | var_ph

            # evaluate expression
            # move tier+dots in expression to underscores (e.g. evt.foo -> evt_foo)
            res = eval(
                expr.replace("dsp.", "dsp_")
                .replace("hit.", "hit_")
                .replace("evt.", "evt_"),
                var,
            )

            # if it is not a nparray it could be a single value
            # expand accordingly
            if not isinstance(res, np.ndarray):
                res = np.full(len(out), res, dtype=type(res))

        # get sub evt based query condition if needed
        if isinstance(qry, str):
            qry_lst = re.findall(r"(hit|dsp).([a-zA-Z_$][\w$]*)", qry)
            qry_var = find_parameters(f_hit, f_dsp, ch, idx_ch, qry_lst)
            limarr = eval(qry.replace("dsp.", "dsp_").replace("hit.", "hit_"), qry_var)

        # or forward the array
        elif isinstance(qry, np.ndarray):
            limarr = qry

        # if no condition, it must be true
        else:
            limarr = np.ones(len(res)).astype(bool)

        # append to out according to mode == tot
        if res.dtype == bool:
            res = res.astype(int)
        if limarr.dtype != bool:
            limarr = limarr.astype(bool)
        out[idx_ch] = np.where(limarr, res + out[idx_ch], out[idx_ch])

    return {"values": out}


def evaluate_to_any(
    idx: np.ndarray,
    ids: np.ndarray,
    f_hit: str,
    f_dsp: str,
    chns: list,
    expr: str,
    exprl: list,
    qry: str | np.ndarray,
    nrows: int,
    var_ph: dict = None,
    defv=np.nan,
) -> dict:
    # define dimension of output array
    out = np.full(nrows, defv, dtype=type(defv))

    for ch in chns:
        # get index list for this channel to be loaded
        idx_ch = idx[ids == int(ch[2:])]

        if "tcm.array_id" == expr:
            res = np.full(len(out), int(ch[2:]), dtype=int)
        else:
            # find fields in either dsp, hit
            var = find_parameters(f_hit, f_dsp, ch, idx_ch, exprl) | var_ph

            # evaluate expression
            # move tier+dots in expression to underscores (e.g. evt.foo -> evt_foo)
            res = eval(
                expr.replace("dsp.", "dsp_")
                .replace("hit.", "hit_")
                .replace("evt.", "evt_"),
                var,
            )

            # if it is not a nparray it could be a single value
            # expand accordingly
            if not isinstance(res, np.ndarray):
                res = np.full(len(out), res, dtype=type(res))

        # get sub evt based query condition if needed
        if isinstance(qry, str):
            qry_lst = re.findall(r"(hit|dsp).([a-zA-Z_$][\w$]*)", qry)
            qry_var = find_parameters(f_hit, f_dsp, ch, idx_ch, qry_lst)
            limarr = eval(qry.replace("dsp.", "dsp_").replace("hit.", "hit_"), qry_var)

        # or forward the array
        elif isinstance(qry, np.ndarray):
            limarr = qry

        # if no condition, it must be true
        else:
            limarr = np.ones(len(res)).astype(bool)

        # append to out according to mode == any
        if res.dtype != bool:
            res = res.astype(bool)
        if limarr.dtype != bool:
            limarr = limarr.astype(bool)
        out[idx_ch] = out[idx_ch] | (res & limarr)

    return {"values": out}


def evaluate_to_all(
    idx: np.ndarray,
    ids: np.ndarray,
    f_hit: str,
    f_dsp: str,
    chns: list,
    expr: str,
    exprl: list,
    qry: str | np.ndarray,
    nrows: int,
    var_ph: dict = None,
    defv=np.nan,
) -> dict:
    # define dimension of output array
    out = np.full(nrows, defv, dtype=type(defv))

    for ch in chns:
        # get index list for this channel to be loaded
        idx_ch = idx[ids == int(ch[2:])]

        if "tcm.array_id" == expr:
            res = np.full(len(out), int(ch[2:]), dtype=int)
        else:
            # find fields in either dsp, hit
            var = find_parameters(f_hit, f_dsp, ch, idx_ch, exprl) | var_ph

            # evaluate expression
            res = eval(
                expr.replace("dsp.", "dsp_")
                .replace("hit.", "hit_")
                .replace("evt.", "evt_"),
                var,
            )

            # if it is not a nparray it could be a single value
            # expand accordingly
            if not isinstance(res, np.ndarray):
                res = np.full(len(out), res, dtype=type(res))

        # get sub evt based query condition if needed
        if isinstance(qry, str):
            qry_lst = re.findall(r"(hit|dsp).([a-zA-Z_$][\w$]*)", qry)
            qry_var = find_parameters(f_hit, f_dsp, ch, idx_ch, qry_lst)
            limarr = eval(qry.replace("dsp.", "dsp_").replace("hit.", "hit_"), qry_var)

        # or forward the array
        elif isinstance(qry, np.ndarray):
            limarr = qry

        # if no condition, it must be true
        else:
            limarr = np.ones(len(res)).astype(bool)

        # append to out according to mode == all
        if res.dtype != bool:
            res = res.astype(bool)
        if limarr.dtype != bool:
            limarr = limarr.astype(bool)
        out[idx_ch] = out[idx_ch] & res & limarr

    return {"values": out}


def evaluate_at_channel(
    idx: np.ndarray,
    ids: np.ndarray,
    f_hit: str,
    f_dsp: str,
    chns: list,
    expr: str,
    exprl: list,
    ch_comp: Array,
    var_ph: dict = None,
    defv=np.nan,
) -> dict:
    out = np.full(len(ch_comp), defv, dtype=type(defv))

    for ch in chns:
        # get index list for this channel to be loaded
        idx_ch = idx[ids == int(ch[2:])]

        if "tcm.array_id" == expr:
            res = np.full(len(out), int(ch[2:]), dtype=int)
        else:
            # find fields in either dsp, hit
            var = find_parameters(f_hit, f_dsp, ch, idx_ch, exprl) | var_ph

            # evaluate expression
            res = eval(
                expr.replace("dsp.", "dsp_")
                .replace("hit.", "hit_")
                .replace("evt.", "evt_"),
                var,
            )

            # if it is not a nparray it could be a single value
            # expand accordingly
            if not isinstance(res, np.ndarray):
                res = np.full(len(out), res, dtype=type(res))

        out[idx_ch] = np.where(int(ch[2:]) == ch_comp.nda, res, out[idx_ch])

    return {"values": out}


def evaluate_at_channel_vov(
    idx: np.ndarray,
    ids: np.ndarray,
    f_hit: str,
    f_dsp: str,
    expr: str,
    exprl: list,
    ch_comp: VectorOfVectors,
    var_ph: dict = None,
) -> dict:
    # blow up vov to aoesa
    out = ch_comp.to_aoesa().nda

    chns = np.unique(out[~np.isnan(out)]).astype(int)

    for ch in chns:
        # get index list for this channel to be loaded
        idx_ch = idx[ids == ch]

        if "tcm.array_id" == expr:
            res = np.full(len(out), int(ch[2:]), dtype=int)
        else:
            # find fields in either dsp, hit
            var = find_parameters(f_hit, f_dsp, f"ch{ch}", idx_ch, exprl) | var_ph

            # evaluate expression
            res = eval(
                expr.replace("dsp.", "dsp_")
                .replace("hit.", "hit_")
                .replace("evt.", "evt_"),
                var,
            )

            # if it is not a nparray it could be a single value
            # expand accordingly
            if not isinstance(res, np.ndarray):
                res = np.full(len(out), res, dtype=type(res))

        # see in which events the current channel is present
        mask = (out == ch).any(axis=1)
        out[out == ch] = res[mask]

    # ok now implode the table again
    out = VectorOfVectors(
        flattened_data=out.flatten()[~np.isnan(out.flatten())].astype(res.dtype),
        cumulative_length=np.cumsum(np.count_nonzero(~np.isnan(out), axis=1)),
    )
    return {"values": out, "channels": ch_comp}


def evaluate_to_vector(
    idx: np.ndarray,
    ids: np.ndarray,
    f_hit: str,
    f_dsp: str,
    chns: list,
    expr: str,
    exprl: list,
    qry: str | np.ndarray,
    nrows: int,
    var_ph: dict = None,
) -> dict:
    """
    Allows the evaluation as a vector of vectors.
    Returns a dictionary of values: VoV of requested values
    and channels: VoV of same dimensions with requested channel_id
    """
    # raise NotImplementedError

    # define dimension of output array
    out = np.full((nrows, len(chns)), np.nan)
    out_chs = np.full((nrows, len(chns)), np.nan)

    i = 0
    for ch in chns:
        # get index list for this channel to be loaded
        idx_ch = idx[ids == int(ch[2:])]

        if "tcm.array_id" == expr:
            res = np.full(len(out), int(ch[2:]), dtype=int)
        else:
            # find fields in either dsp, hit
            var = find_parameters(f_hit, f_dsp, ch, idx_ch, exprl) | var_ph

            # evaluate expression
            res = eval(
                expr.replace("dsp.", "dsp_")
                .replace("hit.", "hit_")
                .replace("evt.", "evt_"),
                var,
            )

            # if it is not a nparray it could be a single value
            # expand accordingly
            if not isinstance(res, np.ndarray):
                res = np.full(len(out), res, dtype=type(res))

        # get sub evt based query condition if needed
        if isinstance(qry, str):
            qry_lst = re.findall(r"(hit|dsp).([a-zA-Z_$][\w$]*)", qry)
            qry_var = find_parameters(f_hit, f_dsp, ch, idx_ch, qry_lst)
            limarr = eval(qry.replace("dsp.", "dsp_").replace("hit.", "hit_"), qry_var)

        # or forward the array
        elif isinstance(qry, np.ndarray):
            limarr = qry

        # if no condition, it must be true
        else:
            limarr = np.ones(len(res)).astype(bool)

        if limarr.dtype != bool:
            limarr = limarr.astype(bool)
        # append to out according to mode == vov
        out[:, i][limarr] = res[limarr]
        out_chs[:, i][limarr] = int(ch[2:])

        i += 1

    # This can be smarter
    # shorten to vov (FUTURE: replace with awkward)
    out = VectorOfVectors(
        flattened_data=out.flatten()[~np.isnan(out.flatten())],
        cumulative_length=np.cumsum(np.count_nonzero(~np.isnan(out), axis=1)),
    )
    out_chs = VectorOfVectors(
        flattened_data=out_chs.flatten()[~np.isnan(out_chs.flatten())].astype(int),
        cumulative_length=np.cumsum(np.count_nonzero(~np.isnan(out_chs), axis=1)),
    )

    return {"values": out, "channels": out_chs}


def build_evt(
    f_tcm: str,
    f_dsp: str,
    f_hit: str,
    f_evt: str,
    evt_config: str | dict,
    wo_mode: str = "write_safe",
    group: str = "/evt/",
    tcm_group: str = "/hardware_tcm_1/",
) -> None:
    """
    Transform data from the hit and dsp levels which a channel sorted
    to a event sorted data format

    Parameters
    ----------
    f_tcm
        input LH5 file of the tcm level
    f_dsp
        input LH5 file of the dsp level
    f_hit
        input LH5 file of the hit level

    f_evt
        name of the output file
    evt_config
        name of JSON file or dict defining evt fields. Channel lists can be defined by the user or by using the keyword "meta" followed by the system (geds/spms) and the usability (on,no_psd,ac,off) separated by underscores (e.g. "meta_geds_on") in the "channels" dictionary. The "operations" dictionary defines the fields (name=key), where "channels" specifies the channels used to for this field (either a string or a list of strings), "mode" defines how the channels should be combined (see evaluate_expression). For first/last modes a "get_ch" flag can be defined, if true an additional field with the sufix "_id" is returned containing the rawid of the respective value in the field without the suffix. "expression" defnies the mathematical/special function to apply (see evaluate_expression), "parameters" defines any other parameter used in expression. For example:

        .. code-block::json

            {
                "channels": {
                    "geds_on": "meta_geds_on",
                    "geds_no_psd": "meta_geds_no_psd",
                    "geds_ac": "meta_geds_ac",
                    "spms_on": "meta_spms_on",
                    "pulser": "PULS01",
                    "baseline": "BSLN01",
                    "muon": "MUON01",
                    "ts_master":"S060"
                },
                "operations": {
                    "energy":{
                        "channels": ["geds_on","geds_no_psd","geds_ac"],
                        "mode": "first>25",
                        "get_ch": true,
                        "expression": "cuspEmax_ctc_cal",
                        "initial": "np.nan"
                    },
                    "energy_on":{
                        "channels": ["geds_on"],
                        "mode": "vov>25",
                        "get_ch": true,
                        "expression": "cuspEmax_ctc_cal"
                    },
                    "aoe":{
                        "channels": ["geds_on"],
                        "mode": "energy_id",
                        "expression": "AoE_Classifier",
                        "initial": "np.nan"
                    },
                    "is_muon_tagged":{
                        "channels": "muon",
                        "mode": "any",
                        "expression": "wf_max>a",
                        "parameters": {"a":15100},
                        "initial": false
                    },
                    "multiplicity":{
                        "channels":  ["geds_on","geds_no_psd","geds_ac"],
                        "mode": "tot",
                        "expression": "cuspEmax_ctc_cal > a",
                        "parameters": {"a":25},
                        "initial": 0
                    },
                    "lar_energy":{
                        "channels": "spms_on",
                        "mode": "func",
                        "expression": "modules.spm.get_energy(0.5,t0,48000,1000,5000)"
                    }
                }
            }

    wo_mode
        writing mode
    group
        lh5 root group name
    tcm_group
        lh5 root group in tcm file
    """

    lstore = store.LH5Store()
    tbl_cfg = evt_config
    if not isinstance(tbl_cfg, (str, dict)):
        raise TypeError()
    if isinstance(tbl_cfg, str):
        with open(tbl_cfg) as f:
            tbl_cfg = json.load(f)

    if "channels" not in tbl_cfg.keys():
        raise ValueError("channel field needs to be specified in the config")
    if "operations" not in tbl_cfg.keys():
        raise ValueError("operations field needs to be specified in the config")

    # create channel list according to config
    # This can be either read from the meta data
    # or a list of channel names
    log.debug("Creating channel dictionary")

    chns = {}

    for k, v in tbl_cfg["channels"].items():
        if isinstance(v, dict):
            # it is a meta module. module_name must exist
            if "module" not in v.keys():
                raise ValueError(
                    "Need module_name to load channel via a meta data module"
                )

            attr = {}
            # the time_key argument is set to the time key of the DSP file
            # in case it is not provided by the config
            if "time_key" not in v.keys():
                attr["time_key"] = re.search(r"\d{8}T\d{6}Z", f_dsp).group(0)

            # if "None" do None
            elif "None" == v["time_key"]:
                attr["time_key"] = None

            # load module
            p, m = v["module"].rsplit(".", 1)
            met = getattr(import_module(p, package=__package__), m)
            chns[k] = met(v | attr)

        elif isinstance(v, str):
            chns[k] = [v]

        elif isinstance(v, list):
            chns[k] = [e for e in v]

    nrows = len(
        store.load_nda(f_tcm, ["cumulative_length"], tcm_group)["cumulative_length"]
    )
    log.info(
        f"Applying {len(tbl_cfg['operations'].keys())} operations to key {f_tcm.split('-')[-2]}"
    )
    for k, v in tbl_cfg["operations"].items():
        log.debug("Processing field" + k)

        # if mode not defined in operation, it can only be an operation on the evt level.
        if "aggregation_mode" not in v.keys():
            exprl = re.findall(r"(evt).([a-zA-Z_$][\w$]*)", v["expression"])
            var = {}
            if os.path.exists(f_evt):
                var = load_vars_to_nda(f_evt, "", exprl)

            if "parameters" in v.keys():
                var = var | v["parameters"]
            res = eval(v["expression"].replace("evt.", "evt_"), var)

            # now check what dimension we have after the evaluation
            if len(res.shape) == 1:
                res = Array(res)
            elif len(res.shape) == 2:
                res = VectorOfVectors(
                    flattened_data=res.flatten()[~np.isnan(res.flatten())],
                    cumulative_length=np.cumsum(
                        np.count_nonzero(~np.isnan(res), axis=1)
                    ),
                )
            else:
                raise NotImplementedError(
                    f"Currently only 2d formats are supported, the evaluated array has the dimension {res.shape}"
                )

            lstore.write_object(
                obj=res,
                name=group + k,
                lh5_file=f_evt,
                wo_mode=wo_mode,
            )

        # Else we build the event entry
        else:
            if "channels" not in v.keys():
                chns_e = []
            elif isinstance(v["channels"], str):
                chns_e = chns[v["channels"]]
            elif isinstance(v["channels"], list):
                chns_e = list(
                    itertools.chain.from_iterable([chns[e] for e in v["channels"]])
                )

            pars, qry, defaultv = None, None, np.nan
            if "parameters" in v.keys():
                pars = v["parameters"]
            if "query" in v.keys():
                qry = v["query"]
            if "initial" in v.keys() and not v["initial"] == "np.nan":
                defaultv = v["initial"]

            result = evaluate_expression(
                f_tcm,
                f_evt,
                f_hit,
                f_dsp,
                chns_e,
                v["aggregation_mode"],
                v["expression"],
                nrows,
                group,
                pars,
                qry,
                defaultv,
            )

            obj = result["values"]
            if isinstance(obj, np.ndarray):
                obj = Array(result["values"])
            lstore.write_object(
                obj=obj,
                name=group + k,
                lh5_file=f_evt,
                wo_mode=wo_mode,
            )

    log.info("Done")


def skim_evt(
    f_evt: str,
    expression: str,
    params: dict = None,
    f_out: str = None,
    wo_mode="n",
    evt_group="/evt/",
) -> None:
    """
    Skimms events from a evt file which are fullfling the expression, discards all other events.

    Parameters
    ----------
    f_evt
        input LH5 file of the evt level
    expression
        skimming expression. Can contain variabels from event file or from the params dictionary.
    f_out
        output LH5 file. Can be None if wo_mode is set to overwrite f_evt.
    wo_mode
        Write mode: "o"/"overwrite" overwrites f_evt. "n"/"new" writes to a new file specified in f_out.
    evt_group
        lh5 root group of the evt file
    """

    if wo_mode not in ["o", "overwrite", "n", "new"]:
        raise ValueError(
            wo_mode
            + " is a invalid writing mode. Valid options are: 'o', 'overwrite','n','new'"
        )
    lstore = store.LH5Store()
    fields = store.ls(f_evt, evt_group)
    nrows = lstore.read_n_rows(fields[0], f_evt)
    # load fields in expression
    exprl = re.findall(r"[a-zA-Z_$][\w$]*", expression)
    var = {}

    flds = [
        e.split("/")[-1]
        for e in store.ls(f_evt, evt_group)
        if e.split("/")[-1] in exprl
    ]
    var = {e: lstore.read_object(evt_group + e, f_evt)[0] for e in flds}

    # to make any operations to VoVs we have to blow it up to a table (future change to more intelligant way)
    arr_keys = []
    for key, value in var.items():
        if isinstance(value, VectorOfVectors):
            var[key] = value.to_aoesa().nda
        elif isinstance(value, Array):
            var[key] = value.nda
            arr_keys.append(key)

    # now we also need to set dimensions if we have an expression
    # consisting of a mix of VoV and Arrays
    if len(arr_keys) > 0 and not set(arr_keys) == set(var.keys()):
        for key in arr_keys:
            var[key] = var[key][:, None]

    if params is not None:
        var = var | params
    res = eval(expression, var)

    if res.shape != (nrows,):
        raise ValueError(
            f"The expression must result to 1D with length = event number. Current shape is {res.shape}"
        )

    res = res.astype(bool)
    idx_list = np.arange(nrows, dtype=int)[res]

    of = f_out
    if wo_mode in ["o", "overwrite"]:
        of = f_evt
    of_tmp = of.replace(of.split("/")[-1], ".tmp_" + of.split("/")[-1])

    for fld in fields:
        ob, _ = lstore.read_object(fld, f_evt, idx=idx_list)
        lstore.write_object(
            obj=ob,
            name=fld,
            lh5_file=of_tmp,
            wo_mode="o",
        )

    if os.path.exists(of):
        os.remove(of)
    os.rename(of_tmp, of)
