"""
This module implements routines to build the `evt` tier.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import random
import re
from importlib import import_module

import numpy as np
from lgdo import Array, VectorOfVectors, lh5
from lgdo.lh5 import LH5Store
from numpy.typing import NDArray

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
    chns_rm: list,
    mode: str,
    expr: str,
    nrows: int,
    para: dict = None,
    qry: str = None,
    defv: bool | int | float = np.nan,
    sorter: str = None,
) -> dict:
    """Evaluates the expression defined by the user across all channels
    according to the mode.

    Parameters
    ----------
    f_tcm
       path to `tcm` tier file.
    f_evt
       path to `evt` tier file.
    f_hit
       path to `hit` tier file.
    f_dsp
       path to `dsp` tier file.
    chns
       list of channel names across which expression gets evaluated (form:
       ``ch<rawid>``).
    chns_rm
       list of channels which get set to default value during evaluation. In
       function mode they are removed entirely (form: ``ch<rawid>``)
    mode
       The mode determines how the event entry is calculated across channels.
       Options are:

       - ``first_at:sorter``: aggregates across channels by returning the
         expression of the channel with smallest value of sorter.
       - ``last_at``: aggregates across channels by returning the expression of
         the channel with largest value of sorter.
       - ``sum``: aggregates by summation.
       - ``any``: aggregates by logical or.
       - ``all``: aggregates by logical and.
       - ``keep_at:ch_field``: aggregates according to passed ch_field
       - ``gather``: Channels are not combined, but result saved as
         :class:`.VectorOfVectors`.

    qry
       a query that can mask the aggregation.
    expr
       the expression. That can be any mathematical equation/comparison. If
       `mode` is ``function``, the expression needs to be a special processing
       function defined in modules (e.g. :func:`.modules.spm.get_energy`). In
       the expression parameters from either hit, dsp, evt tier (from
       operations performed before this one! Dictionary operations order
       matters), or from the ``parameters`` field can be used.
    nrows
       number of rows to be processed.
    para
       dictionary of parameters defined in the ``parameters`` field in the
       configuration dictionary.
    defv
       default value of evaluation.
    sorter
       can be used to sort vector outputs according to sorter expression (see
       :func:`evaluate_to_vector`).
    """

    store = LH5Store()

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
        params = [f_hit, f_dsp, f_tcm, [x for x in chns if x not in chns_rm]] + [
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
        ids = store.read("hardware_tcm_1/array_id", f_tcm)[0].view_as("np")
        idx = store.read("hardware_tcm_1/array_idx", f_tcm)[0].view_as("np")

        # switch through modes
        if (
            os.path.exists(f_evt)
            and "keep_at:" == mode[:8]
            and "evt." == mode[8:][:4]
            and mode[8:].split(".")[-1]
            in [e.split("/")[-1] for e in lh5.ls(f_evt, "/evt/")]
        ):
            ch_comp, _ = store.read(mode[8:].replace(".", "/"), f_evt)
            if isinstance(ch_comp, Array):
                return evaluate_at_channel(
                    idx,
                    ids,
                    f_hit,
                    f_dsp,
                    chns_rm,
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
                    chns_rm,
                    var_ph,
                    defv,
                )
            else:
                raise NotImplementedError(
                    type(ch_comp)
                    + " not supported (only Array and VectorOfVectors are supported)"
                )

        elif "first_at:" in mode:
            sorter = tuple(
                re.findall(
                    r"(evt|hit|dsp).([a-zA-Z_$][\w$]*)", mode.split("first_at:")[-1]
                )[0]
            )
            return evaluate_to_first(
                idx,
                ids,
                f_hit,
                f_dsp,
                chns,
                chns_rm,
                expr,
                exprl,
                qry_mask,
                nrows,
                sorter,
                var_ph,
                defv,
            )
        elif "last_at:" in mode:
            sorter = tuple(
                re.findall(
                    r"(evt|hit|dsp).([a-zA-Z_$][\w$]*)", mode.split("last_at:")[-1]
                )[0]
            )
            return evaluate_to_last(
                idx,
                ids,
                f_hit,
                f_dsp,
                chns,
                chns_rm,
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
                chns_rm,
                expr,
                exprl,
                qry_mask,
                nrows,
                var_ph,
                defv,
            )
        elif "gather" == mode:
            return evaluate_to_vector(
                idx,
                ids,
                f_hit,
                f_dsp,
                chns,
                chns_rm,
                expr,
                exprl,
                qry_mask,
                nrows,
                var_ph,
                defv,
                sorter,
            )
        elif "any" == mode:
            return evaluate_to_any(
                idx,
                ids,
                f_hit,
                f_dsp,
                chns,
                chns_rm,
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
                chns_rm,
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
    idx_ch: NDArray,
    exprl: list,
) -> dict:
    """Wraps :func:`load_vars_to_nda` to return parameters from `hit` and `dsp`
    tiers.

    Parameters
    ----------
    f_hit
       path to `hit` tier file.
    f_dsp
       path to `dsp` tier file.
    ch
       "rawid" in the tiers.
    idx_ch
       index array of entries to be read from files.
    exprl
       list of tuples ``(tier, field)`` to be found in the `hit/dsp` tiers.
    """

    # find fields in either dsp, hit
    var = load_vars_to_nda(f_hit, ch, exprl, idx_ch)
    dsp_dic = load_vars_to_nda(f_dsp, ch, exprl, idx_ch)

    return dsp_dic | var


def load_vars_to_nda(f: str, group: str, exprl: list, idx: NDArray = None) -> dict:
    """Maps parameter expressions to parameters if found in `f`.
    Blows up :class:`.VectorOfVectors` to :class:`.ArrayOfEqualSizedArrays`.

    Parameters
    ----------
    f
       path to a LGDO file.
    group
       additional group in `f`.
    idx
       index array of entries to be read from files.
    exprl
       list of parameter-tuples ``(root_group, field)`` to be found in `f`.
    """

    store = LH5Store()
    var = {
        f"{e[0]}_{e[1]}": store.read(
            f"{group.replace('/','')}/{e[0]}/{e[1]}",
            f,
            idx=idx,
        )[0]
        for e in exprl
        if e[1]
        in [x.split("/")[-1] for x in lh5.ls(f, f"{group.replace('/','')}/{e[0]}/")]
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


def get_data_at_channel(
    ch: str,
    idx_ch: NDArray,
    expr: str,
    exprl: list,
    var_ph: dict,
    is_evaluated: bool,
    f_hit: str,
    f_dsp: str,
    outsize: int,
    defv,
) -> np.ndarray:
    """Evaluates an expression and returns the result.

    Parameters
    ----------
    ch
       "rawid" of channel to be evaluated.
    idx_ch
       array of indices to be evaluated.
    expr
       expression to be evaluated.
    exprl
       list of parameter-tuples ``(root_group, field)`` found in the expression.
    var_ph
       dict of additional parameters that are not channel dependent.
    is_evaluated
       if false, the expression does not get evaluated but an array of default
       values is returned.
    f_hit
       path to `hit` tier file.
    f_dsp
       path to `dsp` tier file.
    outsize
       size of the return array.
    defv
       default value.
    """

    if not is_evaluated:
        res = np.full(outsize, defv, dtype=type(defv))
    elif "tcm.array_id" == expr:
        res = np.full(outsize, int(ch[2:]), dtype=int)
    else:
        var = find_parameters(f_hit, f_dsp, ch, idx_ch, exprl)

        if var_ph is not None:
            var = var | var_ph

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
        res = np.full(outsize, res, dtype=type(res))

    return res


def get_mask_from_query(
    qry: str | NDArray,
    length: int,
    ch: str,
    idx_ch: NDArray,
    f_hit: str,
    f_dsp: str,
) -> np.ndarray:
    """Evaluates a query expression and returns a mask accordingly.

    Parameters
    ----------
    qry
       query expression.
    length
       length of the return mask.
    ch
       "rawid" of channel to be evaluated.
    idx_ch
       array of indices to be evaluated.
    f_hit
       path to `hit` tier file.
    f_dsp
       path to `dsp` tier file.
    """

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
        limarr = np.ones(length).astype(bool)

    if limarr.dtype != bool:
        limarr = limarr.astype(bool)

    return limarr


def evaluate_to_first(
    idx: NDArray,
    ids: NDArray,
    f_hit: str,
    f_dsp: str,
    chns: list,
    chns_rm: list,
    expr: str,
    exprl: list,
    qry: str | NDArray,
    nrows: int,
    sorter: tuple,
    var_ph: dict = None,
    defv: bool | int | float = np.nan,
) -> dict:
    """Aggregates across channels by returning the expression of the channel
    with smallest value of `sorter`.

    Parameters
    ----------
    idx
       `tcm` index array.
    ids
       `tcm` id array.
    f_hit
       path to `hit` tier file.
    f_dsp
       path to `dsp` tier file.
    chns
       list of channels to be aggregated.
    chns_rm
       list of channels to be skipped from evaluation and set to default value.
    expr
       expression string to be evaluated.
    exprl
       list of `dsp/hit/evt` parameter tuples in expression ``(tier, field)``.
    qry
       query expression to mask aggregation.
    nrows
       length of output array.
    sorter
       tuple of field in `hit/dsp/evt` tier to evaluate ``(tier, field)``.
    var_ph
       dictionary of `evt` and additional parameters and their values.
    defv
       default value.
    """

    # define dimension of output array
    out = np.full(nrows, defv, dtype=type(defv))
    out_chs = np.zeros(len(out), dtype=int)
    outt = np.zeros(len(out))

    store = LH5Store()

    for ch in chns:
        # get index list for this channel to be loaded
        idx_ch = idx[ids == int(ch[2:])]

        # evaluate at channel
        res = get_data_at_channel(
            ch,
            idx_ch,
            expr,
            exprl,
            var_ph,
            ch not in chns_rm,
            f_hit,
            f_dsp,
            len(out),
            defv,
        )

        # get mask from query
        limarr = get_mask_from_query(qry, len(res), ch, idx_ch, f_hit, f_dsp)

        # append to out according to mode == first
        if ch == chns[0]:
            outt[:] = np.inf

        # find if sorter is in hit or dsp
        t0 = store.read(
            f"{ch}/{sorter[0]}/{sorter[1]}",
            f_hit if "hit" == sorter[0] else f_dsp,
            idx=idx_ch,
        )[0].view_as("np")

        out[idx_ch] = np.where((t0 < outt) & (limarr), res, out[idx_ch])
        out_chs[idx_ch] = np.where((t0 < outt) & (limarr), int(ch[2:]), out_chs[idx_ch])
        outt[idx_ch] = np.where((t0 < outt) & (limarr), t0, outt[idx_ch])

    return {"values": out, "channels": out_chs}


def evaluate_to_last(
    idx: NDArray,
    ids: NDArray,
    f_hit: str,
    f_dsp: str,
    chns: list,
    chns_rm: list,
    expr: str,
    exprl: list,
    qry: str | NDArray,
    nrows: int,
    sorter: tuple,
    var_ph: dict = None,
    defv: bool | int | float = np.nan,
) -> dict:
    """Aggregates across channels by returning the expression of the channel
    with largest value of `sorter`.

    Parameters
    ----------
    idx
       `tcm` index array.
    ids
       `tcm` id array.
    f_hit
       path to `hit` tier file.
    f_dsp
       path to `dsp` tier file.
    chns
       list of channels to be aggregated.
    chns_rm
       list of channels to be skipped from evaluation and set to default value.
    expr
       expression string to be evaluated.
    exprl
       list of dsp/hit/evt parameter tuples in expression ``(tier, field)``.
    qry
       query expression to mask aggregation.
    nrows
       length of output array.
    sorter
       tuple of field in `hit/dsp/evt` tier to evaluate ``(tier, field)``.
    var_ph
       dictionary of `evt` and additional parameters and their values.
    defv
       default value.
    """

    # define dimension of output array
    out = np.full(nrows, defv, dtype=type(defv))
    out_chs = np.zeros(len(out), dtype=int)
    outt = np.zeros(len(out))

    store = LH5Store()

    for ch in chns:
        # get index list for this channel to be loaded
        idx_ch = idx[ids == int(ch[2:])]

        # evaluate at channel
        res = get_data_at_channel(
            ch,
            idx_ch,
            expr,
            exprl,
            var_ph,
            ch not in chns_rm,
            f_hit,
            f_dsp,
            len(out),
            defv,
        )

        # get mask from query
        limarr = get_mask_from_query(qry, len(res), ch, idx_ch, f_hit, f_dsp)

        # append to out according to mode == last
        # find if sorter is in hit or dsp
        t0 = store.read(
            f"{ch}/{sorter[0]}/{sorter[1]}",
            f_hit if "hit" == sorter[0] else f_dsp,
            idx=idx_ch,
        )[0].view_as("np")

        out[idx_ch] = np.where((t0 > outt) & (limarr), res, out[idx_ch])
        out_chs[idx_ch] = np.where((t0 > outt) & (limarr), int(ch[2:]), out_chs[idx_ch])
        outt[idx_ch] = np.where((t0 > outt) & (limarr), t0, outt[idx_ch])

    return {"values": out, "channels": out_chs}


def evaluate_to_tot(
    idx: NDArray,
    ids: NDArray,
    f_hit: str,
    f_dsp: str,
    chns: list,
    chns_rm: list,
    expr: str,
    exprl: list,
    qry: str | NDArray,
    nrows: int,
    var_ph: dict = None,
    defv: bool | int | float = np.nan,
) -> dict:
    """Aggregates by summation across channels.

    Parameters
    ----------
    idx
       tcm index array.
    ids
       tcm id array.
    f_hit
       path to `hit` tier file.
    f_dsp
       path to `dsp` tier file.
    chns
       list of channels to be aggregated.
    chns_rm
       list of channels to be skipped from evaluation and set to default value.
    expr
       expression string to be evaluated.
    exprl
       list of dsp/hit/evt parameter tuples in expression (tier, field).
    qry
       query expression to mask aggregation.
    nrows
       length of output array
    var_ph
       dictionary of evt and additional parameters and their values.
    defv
       default value.
    """

    # define dimension of output array
    out = np.full(nrows, defv, dtype=type(defv))

    for ch in chns:
        # get index list for this channel to be loaded
        idx_ch = idx[ids == int(ch[2:])]

        res = get_data_at_channel(
            ch,
            idx_ch,
            expr,
            exprl,
            var_ph,
            ch not in chns_rm,
            f_hit,
            f_dsp,
            len(out),
            defv,
        )

        # get mask from query
        limarr = get_mask_from_query(qry, len(res), ch, idx_ch, f_hit, f_dsp)

        # append to out according to mode == tot
        if res.dtype == bool:
            res = res.astype(int)

        out[idx_ch] = np.where(limarr, res + out[idx_ch], out[idx_ch])

    return {"values": out}


def evaluate_to_any(
    idx: NDArray,
    ids: NDArray,
    f_hit: str,
    f_dsp: str,
    chns: list,
    chns_rm: list,
    expr: str,
    exprl: list,
    qry: str | NDArray,
    nrows: int,
    var_ph: dict = None,
    defv: bool | int | float = np.nan,
) -> dict:
    """Aggregates by logical or operation across channels. If the expression
    evaluates to a non boolean value it is casted to boolean.

    Parameters
    ----------
    idx
       `tcm` index array.
    ids
       `tcm` id array.
    f_hit
       path to `hit` tier file.
    f_dsp
       path to `dsp` tier file.
    chns
       list of channels to be aggregated.
    chns_rm
       list of channels to be skipped from evaluation and set to default value.
    expr
       expression string to be evaluated.
    exprl
       list of `dsp/hit/evt` parameter tuples in expression ``(tier, field)``.
    qry
       query expression to mask aggregation.
    nrows
       length of output array.
    var_ph
       dictionary of `evt` and additional parameters and their values.
    defv
       default value.
    """

    # define dimension of output array
    out = np.full(nrows, defv, dtype=type(defv))

    for ch in chns:
        # get index list for this channel to be loaded
        idx_ch = idx[ids == int(ch[2:])]

        res = get_data_at_channel(
            ch,
            idx_ch,
            expr,
            exprl,
            var_ph,
            ch not in chns_rm,
            f_hit,
            f_dsp,
            len(out),
            defv,
        )

        # get mask from query
        limarr = get_mask_from_query(qry, len(res), ch, idx_ch, f_hit, f_dsp)

        # append to out according to mode == any
        if res.dtype != bool:
            res = res.astype(bool)

        out[idx_ch] = out[idx_ch] | (res & limarr)

    return {"values": out}


def evaluate_to_all(
    idx: NDArray,
    ids: NDArray,
    f_hit: str,
    f_dsp: str,
    chns: list,
    chns_rm: list,
    expr: str,
    exprl: list,
    qry: str | NDArray,
    nrows: int,
    var_ph: dict = None,
    defv: bool | int | float = np.nan,
) -> dict:
    """Aggregates by logical and operation across channels. If the expression
    evaluates to a non boolean value it is casted to boolean.

    Parameters
    ----------
    idx
       `tcm` index array.
    ids
       `tcm` id array.
    f_hit
       path to `hit` tier file.
    f_dsp
       path to `dsp` tier file.
    chns
       list of channels to be aggregated.
    chns_rm
       list of channels to be skipped from evaluation and set to default value.
    expr
       expression string to be evaluated.
    exprl
       list of `dsp/hit/evt` parameter tuples in expression ``(tier, field)``.
    qry
       query expression to mask aggregation.
    nrows
       length of output array.
    var_ph
       dictionary of evt and additional parameters and their values.
    defv
       default value.
    """

    # define dimension of output array
    out = np.full(nrows, defv, dtype=type(defv))

    for ch in chns:
        # get index list for this channel to be loaded
        idx_ch = idx[ids == int(ch[2:])]

        res = get_data_at_channel(
            ch,
            idx_ch,
            expr,
            exprl,
            var_ph,
            ch not in chns_rm,
            f_hit,
            f_dsp,
            len(out),
            defv,
        )

        # get mask from query
        limarr = get_mask_from_query(qry, len(res), ch, idx_ch, f_hit, f_dsp)

        # append to out according to mode == all
        if res.dtype != bool:
            res = res.astype(bool)

        out[idx_ch] = out[idx_ch] & res & limarr

    return {"values": out}


def evaluate_at_channel(
    idx: NDArray,
    ids: NDArray,
    f_hit: str,
    f_dsp: str,
    chns_rm: list,
    expr: str,
    exprl: list,
    ch_comp: Array,
    var_ph: dict = None,
    defv: bool | int | float = np.nan,
) -> dict:
    """Aggregates by evaluating the expression at a given channel.

    Parameters
    ----------
    idx
       `tcm` index array.
    ids
       `tcm` id array.
    f_hit
       path to `hit` tier file.
    f_dsp
       path to `dsp` tier file.
    chns_rm
       list of channels to be skipped from evaluation and set to default value.
    expr
       expression string to be evaluated.
    exprl
       list of `dsp/hit/evt` parameter tuples in expression ``(tier, field)``.
    ch_comp
       array of rawids at which the expression is evaluated.
    var_ph
       dictionary of `evt` and additional parameters and their values.
    defv
       default value.
    """

    out = np.full(len(ch_comp.nda), defv, dtype=type(defv))

    for ch in np.unique(ch_comp.nda.astype(int)):
        # skip default value
        if f"ch{ch}" not in lh5.ls(f_hit):
            continue
        # get index list for this channel to be loaded
        idx_ch = idx[ids == ch]

        res = get_data_at_channel(
            f"ch{ch}",
            idx_ch,
            expr,
            exprl,
            var_ph,
            f"ch{ch}" not in chns_rm,
            f_hit,
            f_dsp,
            len(out),
            defv,
        )

        out = np.where(ch == ch_comp.nda, res, out)

    return {"values": out}


def evaluate_at_channel_vov(
    idx: NDArray,
    ids: NDArray,
    f_hit: str,
    f_dsp: str,
    expr: str,
    exprl: list,
    ch_comp: VectorOfVectors,
    chns_rm: list,
    var_ph: dict = None,
    defv: bool | int | float = np.nan,
) -> dict:
    """Same as :func:`evaluate_at_channel` but evaluates expression at non
    flat channels :class:`.VectorOfVectors`.

    Parameters
    ----------
    idx
       `tcm` index array.
    ids
       `tcm` id array.
    f_hit
       path to `hit` tier file.
    f_dsp
       path to `dsp` tier file.
    expr
       expression string to be evaluated.
    exprl
       list of `dsp/hit/evt` parameter tuples in expression ``(tier, field)``.
    ch_comp
       array of "rawid"s at which the expression is evaluated.
    chns_rm
       list of channels to be skipped from evaluation and set to default value.
    var_ph
       dictionary of `evt` and additional parameters and their values.
    defv
       default value.
    """

    # blow up vov to aoesa
    out = ch_comp.to_aoesa().nda

    chns = np.unique(out[~np.isnan(out)]).astype(int)
    type_name = None
    for ch in chns:
        # get index list for this channel to be loaded
        idx_ch = idx[ids == ch]
        res = get_data_at_channel(
            f"ch{ch}",
            idx_ch,
            expr,
            exprl,
            var_ph,
            f"ch{ch}" not in chns_rm,
            f_hit,
            f_dsp,
            len(out),
            defv,
        )

        # see in which events the current channel is present
        mask = (out == ch).any(axis=1)
        out[out == ch] = res[mask]

        if ch == chns[0]:
            type_name = res.dtype

    # ok now implode the table again
    out = VectorOfVectors(
        flattened_data=out.flatten()[~np.isnan(out.flatten())].astype(type_name),
        cumulative_length=np.cumsum(np.count_nonzero(~np.isnan(out), axis=1)),
    )
    return {"values": out, "channels": ch_comp}


def evaluate_to_aoesa(
    idx: NDArray,
    ids: NDArray,
    f_hit: str,
    f_dsp: str,
    chns: list,
    chns_rm: list,
    expr: str,
    exprl: list,
    qry: str | NDArray,
    nrows: int,
    var_ph: dict = None,
    defv: bool | int | float = np.nan,
    missv=np.nan,
) -> np.ndarray:
    """Aggregates by returning an :class:`.ArrayOfEqualSizedArrays` of evaluated
    expressions of channels that fulfill a query expression.

    Parameters
    ----------
    idx
       `tcm` index array.
    ids
       `tcm` id array.
    f_hit
       path to `hit` tier file.
    f_dsp
       path to `dsp` tier file.
    chns
       list of channels to be aggregated.
    chns_rm
       list of channels to be skipped from evaluation and set to default value.
    expr
       expression string to be evaluated.
    exprl
       list of `dsp/hit/evt` parameter tuples in expression ``(tier, field)``.
    qry
       query expression to mask aggregation.
    nrows
       length of output :class:`.VectorOfVectors`.
    ch_comp
       array of "rawid"s at which the expression is evaluated.
    var_ph
       dictionary of `evt` and additional parameters and their values.
    defv
       default value.
    missv
       missing value.
    sorter
       sorts the entries in the vector according to sorter expression.
    """
    # define dimension of output array
    out = np.full((nrows, len(chns)), missv)

    i = 0
    for ch in chns:
        # get index list for this channel to be loaded
        idx_ch = idx[ids == int(ch[2:])]

        res = get_data_at_channel(
            ch,
            idx_ch,
            expr,
            exprl,
            var_ph,
            ch not in chns_rm,
            f_hit,
            f_dsp,
            len(out),
            defv,
        )

        # get mask from query
        limarr = get_mask_from_query(qry, len(res), ch, idx_ch, f_hit, f_dsp)

        # append to out according to mode == vov
        out[:, i][limarr] = res[limarr]

        i += 1

    return out


def evaluate_to_vector(
    idx: NDArray,
    ids: NDArray,
    f_hit: str,
    f_dsp: str,
    chns: list,
    chns_rm: list,
    expr: str,
    exprl: list,
    qry: str | NDArray,
    nrows: int,
    var_ph: dict = None,
    defv: bool | int | float = np.nan,
    sorter: str = None,
) -> dict:
    """Aggregates by returning a :class:`.VectorOfVector` of evaluated
    expressions of channels that fulfill a query expression.

    Parameters
    ----------
    idx
       `tcm` index array.
    ids
       `tcm` id array.
    f_hit
       path to `hit` tier file.
    f_dsp
       path to `dsp` tier file.
    chns
       list of channels to be aggregated.
    chns_rm
       list of channels to be skipped from evaluation and set to default value.
    expr
       expression string to be evaluated.
    exprl
       list of `dsp/hit/evt` parameter tuples in expression ``(tier, field)``.
    qry
       query expression to mask aggregation.
    nrows
       length of output :class:`.VectorOfVectors`.
    ch_comp
       array of "rawids" at which the expression is evaluated.
    var_ph
       dictionary of `evt` and additional parameters and their values.
    defv
       default value.
    sorter
       sorts the entries in the vector according to sorter expression.
       ``ascend_by:<hit|dsp.field>`` results in an vector ordered ascending,
       ``decend_by:<hit|dsp.field>`` sorts descending.
    """
    out = evaluate_to_aoesa(
        idx,
        ids,
        f_hit,
        f_dsp,
        chns,
        chns_rm,
        expr,
        exprl,
        qry,
        nrows,
        var_ph,
        defv,
        np.nan,
    )

    # if a sorter is given sort accordingly
    if sorter is not None:
        md, fld = sorter.split(":")
        s_val = evaluate_to_aoesa(
            idx,
            ids,
            f_hit,
            f_dsp,
            chns,
            chns_rm,
            fld,
            [tuple(fld.split("."))],
            None,
            nrows,
        )
        if "ascend_by" == md:
            out[np.arange(len(out))[:, None], np.argsort(s_val)]

        elif "descend_by" == md:
            out[np.arange(len(out))[:, None], np.argsort(-s_val)]
        else:
            raise ValueError(
                "sorter values can only have 'ascend_by' or 'descend_by' prefixes"
            )

    # This can be smarter
    # shorten to vov (FUTURE: replace with awkward)
    out = VectorOfVectors(
        flattened_data=out.flatten()[~np.isnan(out.flatten())],
        cumulative_length=np.cumsum(np.count_nonzero(~np.isnan(out), axis=1)),
    )

    return {"values": out}


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
    """Transform data from the `hit` and `dsp` levels which a channel sorted to a
    event sorted data format.

    Parameters
    ----------
    f_tcm
        input LH5 file of the tcm level.
    f_dsp
        input LH5 file of the dsp level.
    f_hit
        input LH5 file of the hit level.
    f_evt
        name of the output file.
    evt_config
        name of configuration file or dictionary defining event fields. Channel
        lists can be defined by importing a metadata module.

        - ``operations`` defines the fields ``name=key``, where ``channels``
          specifies the channels used to for this field (either a string or a
          list of strings),
        - ``aggregation_mode`` defines how the channels should be combined (see
          :func:`evaluate_expression`).
        - ``expression`` defnies the mathematical/special function to apply
          (see :func:`evaluate_expression`),
        - ``query`` defines an expression to mask the aggregation.
        - ``parameters`` defines any other parameter used in expression.

        For example:

        .. code-block:: json

            {
              "channels": {
                "geds_on": ["ch1084803", "ch1084804", "ch1121600"],
                "spms_on": ["ch1057600", "ch1059201", "ch1062405"],
                "muon": "ch1027202",
              },
              "operations": {
                "energy_id":{
                  "channels": "geds_on",
                  "aggregation_mode": "gather",
                  "query": "hit.cuspEmax_ctc_cal > 25",
                  "expression": "tcm.array_id",
                  "sort": "ascend_by:dsp.tp_0_est"
                },
                "energy":{
                  "aggregation_mode": "keep_at:evt.energy_id",
                  "expression": "hit.cuspEmax_ctc_cal > 25"
                }
                "is_muon_rejected":{
                  "channels": "muon",
                  "aggregation_mode": "any",
                  "expression": "dsp.wf_max>a",
                  "parameters": {"a":15100},
                  "initial": false
                },
                "multiplicity":{
                  "channels":  ["geds_on", "geds_no_psd", "geds_ac"],
                  "aggregation_mode": "sum",
                  "expression": "hit.cuspEmax_ctc_cal > a",
                  "parameters": {"a":25},
                  "initial": 0
                },
                "t0":{
                  "aggregation_mode": "keep_at:evt.energy_id",
                  "expression": "dsp.tp_0_est"
                },
                "lar_energy":{
                  "channels": "spms_on",
                  "aggregation_mode": "function",
                  "expression": ".modules.spm.get_energy(0.5, evt.t0, 48000, 1000, 5000)"
                },
              }
            }

    wo_mode
        writing mode.
    group
        LH5 root group name.
    tcm_group
        LH5 root group in tcm file.
    """
    store = LH5Store()
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
        lh5.load_nda(f_tcm, ["cumulative_length"], tcm_group)["cumulative_length"]
    )
    # nrows = store.read_n_rows(f"{tcm_group}/cumulative_length", f_tcm)
    log.info(
        f"Applying {len(tbl_cfg['operations'].keys())} operations to key {f_tcm.split('-')[-2]}"
    )

    # Define temporary file
    f_evt_tmp = f"{os.path.dirname(f_evt)}/{os.path.basename(f_evt).split('.')[0]}_tmp{random.randrange(9999):04d}.lh5"

    for k, v in tbl_cfg["operations"].items():
        log.debug("Processing field" + k)

        # if mode not defined in operation, it can only be an operation on the evt level.
        if "aggregation_mode" not in v.keys():
            exprl = re.findall(r"(evt).([a-zA-Z_$][\w$]*)", v["expression"])
            var = {}
            if os.path.exists(f_evt_tmp):
                var = load_vars_to_nda(f_evt_tmp, "", exprl)

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

            store.write(
                res,
                group + k,
                f_evt_tmp,
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
            chns_rm = []
            if "exclude_channels" in v.keys():
                if isinstance(v["exclude_channels"], str):
                    chns_rm = chns[v["exclude_channels"]]
                elif isinstance(v["exclude_channels"], list):
                    chns_rm = list(
                        itertools.chain.from_iterable(
                            [chns[e] for e in v["exclude_channels"]]
                        )
                    )

            pars, qry, defaultv, srter = None, None, np.nan, None
            if "parameters" in v.keys():
                pars = v["parameters"]
            if "query" in v.keys():
                qry = v["query"]
            if "initial" in v.keys() and not v["initial"] == "np.nan":
                defaultv = v["initial"]
            if "sort" in v.keys():
                srter = v["sort"]

            result = evaluate_expression(
                f_tcm,
                f_evt_tmp,
                f_hit,
                f_dsp,
                chns_e,
                chns_rm,
                v["aggregation_mode"],
                v["expression"],
                nrows,
                pars,
                qry,
                defaultv,
                srter,
            )

            obj = result["values"]
            if isinstance(obj, np.ndarray):
                obj = Array(result["values"])
            store.write(
                obj,
                group + k,
                f_evt_tmp,
                wo_mode=wo_mode,
            )

    # write output fields into f_evt and delete temporary file
    if "outputs" in tbl_cfg.keys():
        if len(tbl_cfg["outputs"]) < 1:
            log.warning("No output fields specified, no file will be written.")
        for fld in tbl_cfg["outputs"]:
            obj, _ = store.read(group + fld, f_evt_tmp)
            store.write(
                obj,
                group + fld,
                f_evt,
                wo_mode=wo_mode,
            )
    else:
        log.warning("No output fields specified, no file will be written.")

    os.remove(f_evt_tmp)

    log.info("Done")


def skim_evt(
    f_evt: str,
    expression: str,
    params: dict = None,
    f_out: str = None,
    wo_mode="n",
    evt_group="/evt/",
) -> None:
    """Skims events from an `evt` file which are fulfilling the expression,
    discards all other events.

    Parameters
    ----------
    f_evt
        input LH5 file of the `evt` level.
    expression
        skimming expression. Can contain variables from event file or from the
        `params` dictionary.
    f_out
        output LH5 file. Can be ``None`` if `wo_mode` is set to overwrite `f_evt`.
    wo_mode
        Write mode: ``o``/``overwrite`` overwrites f_evt. ``n``/``new`` writes
        to a new file specified in `f_out`.
    evt_group
        LH5 root group of the `evt` file.
    """

    if wo_mode not in ["o", "overwrite", "n", "new"]:
        raise ValueError(
            wo_mode
            + " is a invalid writing mode. Valid options are: 'o', 'overwrite','n','new'"
        )
    store = LH5Store()
    fields = lh5.ls(f_evt, evt_group)
    nrows = store.read_n_rows(fields[0], f_evt)
    # load fields in expression
    exprl = re.findall(r"[a-zA-Z_$][\w$]*", expression)
    var = {}

    flds = [
        e.split("/")[-1] for e in lh5.ls(f_evt, evt_group) if e.split("/")[-1] in exprl
    ]
    var = {e: store.read(evt_group + e, f_evt)[0] for e in flds}

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
            "The expression must result to 1D with length = event number. "
            f"Current shape is {res.shape}"
        )

    res = res.astype(bool)
    idx_list = np.arange(nrows, dtype=int)[res]

    of = f_out
    if wo_mode in ["o", "overwrite"]:
        of = f_evt
    of_tmp = of.replace(of.split("/")[-1], ".tmp_" + of.split("/")[-1])

    for fld in fields:
        ob, _ = store.read(fld, f_evt, idx=idx_list)
        store.write(
            ob,
            fld,
            of_tmp,
            wo_mode="o",
        )

    if os.path.exists(of):
        os.remove(of)
    os.rename(of_tmp, of)
