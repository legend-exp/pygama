"""
This module implements routines to build the `evt` tier.
"""

from __future__ import annotations

import itertools
import json
import logging
import re
from importlib import import_module

import awkward as ak
import numpy as np
from lgdo import Array, ArrayOfEqualSizedArrays, Table, VectorOfVectors, lh5
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
    f_hit: str,
    f_dsp: str,
    chns: list,
    chns_rm: list,
    mode: str,
    expr: str,
    nrows: int,
    table: Table = None,
    para: dict = None,
    qry: str = None,
    defv: bool | int | float = np.nan,
    sorter: str = None,
) -> Array | ArrayOfEqualSizedArrays | VectorOfVectors:
    """Evaluates the expression defined by the user across all channels
    according to the mode.

    Parameters
    ----------
    f_tcm
       path to `tcm` tier file.
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
    table
       table of 'evt' tier data.
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
    if table:
        var_ph = var_ph | {
            e: table[e].view_as("ak")
            for e in table.keys()
            if isinstance(table[e], (Array, ArrayOfEqualSizedArrays, VectorOfVectors))
        }
    if para:
        var_ph = var_ph | para

    if mode == "function":
        # evaluate expression
        func, params = expr.split("(")
        params = (
            params.replace("dsp.", "dsp_").replace("hit.", "hit_").replace("evt.", "")
        )
        params = [f_hit, f_dsp, f_tcm, [x for x in chns if x not in chns_rm]] + [
            num_and_pars(e, var_ph) for e in params[:-1].split(",")
        ]

        # load function dynamically
        p, m = func.rsplit(".", 1)
        met = getattr(import_module(p, package=__package__), m)
        return met(*params)

    else:
        # check if query is either on channel basis or evt basis (and not a mix)
        qry_mask = qry
        if qry is not None:
            if "evt." in qry and ("hit." in qry or "dsp." in qry):
                raise ValueError("Query can't be a mix of evt tier and lower tiers.")

            # if it is an evt query we can evaluate it directly here
            if table and "evt." in qry:
                qry_mask = eval(qry.replace("evt.", ""), table)

        # load TCM data to define an event
        ids = store.read("hardware_tcm_1/array_id", f_tcm)[0].view_as("np")
        idx = store.read("hardware_tcm_1/array_idx", f_tcm)[0].view_as("np")

        # switch through modes
        if (
            table
            and "keep_at:" == mode[:8]
            and "evt." == mode[8:][:4]
            and mode[8:].split(".")[-1] in table.keys()
        ):
            ch_comp = table[mode[8:].replace("evt.", "")]
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
        elif "first_at:" in mode or "last_at:" in mode:
            sorter = tuple(
                re.findall(
                    r"(evt|hit|dsp).([a-zA-Z_$][\w$]*)", mode.split("first_at:")[-1]
                )[0]
            )
            return evaluate_to_first_or_last(
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
                is_first=True if "first_at:" in mode else False,
            )
        elif mode in ["sum", "any", "all"]:
            return evaluate_to_scalar(
                mode,
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
    dsp_flds = [e[1] for e in exprl if e[0] == "dsp"]
    hit_flds = [e[1] for e in exprl if e[0] == "hit"]

    store = LH5Store()
    hit_dict, dsp_dict = {}, {}
    if len(hit_flds) > 0:
        hit_ak = store.read(
            f"{ch.replace('/','')}/hit/", f_hit, field_mask=hit_flds, idx=idx_ch
        )[0].view_as("ak")
        hit_dict = dict(zip(["hit_" + e for e in ak.fields(hit_ak)], ak.unzip(hit_ak)))
    if len(dsp_flds) > 0:
        dsp_ak = store.read(
            f"{ch.replace('/','')}/dsp/", f_dsp, field_mask=dsp_flds, idx=idx_ch
        )[0].view_as("ak")
        dsp_dict = dict(zip(["dsp_" + e for e in ak.fields(dsp_ak)], ak.unzip(dsp_ak)))

    return hit_dict | dsp_dict


def get_data_at_channel(
    ch: str,
    ids: NDArray,
    idx: NDArray,
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
    idx
       `tcm` index array.
    ids
       `tcm` id array.
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

    # get index list for this channel to be loaded
    idx_ch = idx[ids == int(ch[2:])]

    if not is_evaluated:
        res = np.full(outsize, defv, dtype=type(defv))
    elif "tcm.array_id" == expr:
        res = np.full(outsize, int(ch[2:]), dtype=int)
    elif "tcm.index" == expr:
        res = np.where(ids == int(ch[2:]))[0]
    else:
        var = find_parameters(f_hit, f_dsp, ch, idx_ch, exprl)

        if var_ph is not None:
            var = var | var_ph

        # evaluate expression
        # move tier+dots in expression to underscores (e.g. evt.foo -> evt_foo)
        res = eval(
            expr.replace("dsp.", "dsp_").replace("hit.", "hit_").replace("evt.", ""),
            var,
        )

        # in case the expression evaluates to a single value blow it up
        if (not hasattr(res, "__len__")) or (isinstance(res, str)):
            return np.full(outsize, res)

        # the resulting arrays need to be 1D from the operation,
        # this can only change once we support larger than two dimensional LGDOs
        # ak.to_numpy() raises error if array not regular
        res = ak.to_numpy(res, allow_missing=False)

        # in this method only 1D values are allowed
        if res.ndim > 1:
            raise ValueError(
                f"expression '{expr}' must return 1D array. If you are using VectorOfVectors or ArrayOfEqualSizedArrays, use awkward reduction functions to reduce the dimension"
            )

    return res


def get_mask_from_query(
    qry: str | NDArray,
    length: int,
    ch: str,
    ids: NDArray,
    idx: NDArray,
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
    idx
       `tcm` index array.
    ids
       `tcm` id array.
    f_hit
       path to `hit` tier file.
    f_dsp
       path to `dsp` tier file.
    """
    # get index list for this channel to be loaded
    idx_ch = idx[ids == int(ch[2:])]

    # get sub evt based query condition if needed
    if isinstance(qry, str):
        qry_lst = re.findall(r"(hit|dsp).([a-zA-Z_$][\w$]*)", qry)
        qry_var = find_parameters(f_hit, f_dsp, ch, idx_ch, qry_lst)
        limarr = eval(qry.replace("dsp.", "dsp_").replace("hit.", "hit_"), qry_var)

        # in case the expression evaluates to a single value blow it up
        if (not hasattr(limarr, "__len__")) or (isinstance(limarr, str)):
            return np.full(len(idx_ch), limarr)

        limarr = ak.to_numpy(limarr, allow_missing=False)
        if limarr.ndim > 1:
            raise ValueError(
                f"query '{qry}' must return 1D array. If you are using VectorOfVectors or ArrayOfEqualSizedArrays, use awkward reduction functions to reduce the dimension"
            )

    # or forward the array
    elif isinstance(qry, np.ndarray):
        limarr = qry

    # if no condition, it must be true
    else:
        limarr = np.ones(length).astype(bool)

    # explicit cast to bool
    if limarr.dtype != bool:
        limarr = limarr.astype(bool)

    return limarr


def evaluate_to_first_or_last(
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
    is_first: bool = True,
) -> Array:
    """Aggregates across channels by returning the expression of the channel
    with value of `sorter`.

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
    is_first
       defines if sorted by smallest or largest value of `sorter`
    """

    # define dimension of output array
    out = np.full(nrows, defv, dtype=type(defv))
    outt = np.zeros(len(out))

    store = LH5Store()

    for ch in chns:
        # get index list for this channel to be loaded
        idx_ch = idx[ids == int(ch[2:])]

        # evaluate at channel
        res = get_data_at_channel(
            ch,
            ids,
            idx,
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
        limarr = get_mask_from_query(qry, len(res), ch, ids, idx, f_hit, f_dsp)

        # find if sorter is in hit or dsp
        t0 = store.read(
            f"{ch}/{sorter[0]}/{sorter[1]}",
            f_hit if "hit" == sorter[0] else f_dsp,
            idx=idx_ch,
        )[0].view_as("np")

        if t0.ndim > 1:
            raise ValueError(f"sorter '{sorter[0]}/{sorter[1]}' must be a 1D array")

        if is_first:
            if ch == chns[0]:
                outt[:] = np.inf

            out[idx_ch] = np.where((t0 < outt) & (limarr), res, out[idx_ch])
            outt[idx_ch] = np.where((t0 < outt) & (limarr), t0, outt[idx_ch])

        else:
            out[idx_ch] = np.where((t0 > outt) & (limarr), res, out[idx_ch])
            outt[idx_ch] = np.where((t0 > outt) & (limarr), t0, outt[idx_ch])

    return Array(nda=out)


def evaluate_to_scalar(
    mode: str,
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
) -> Array:
    """Aggregates by summation across channels.

    Parameters
    ----------
    mode
       aggregation mode.
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
            ids,
            idx,
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
        limarr = get_mask_from_query(qry, len(res), ch, ids, idx, f_hit, f_dsp)

        # switch through modes
        if "sum" == mode:
            if res.dtype == bool:
                res = res.astype(int)
            out[idx_ch] = np.where(limarr, res + out[idx_ch], out[idx_ch])
        if "any" == mode:
            if res.dtype != bool:
                res = res.astype(bool)
            out[idx_ch] = out[idx_ch] | (res & limarr)
        if "all" == mode:
            if res.dtype != bool:
                res = res.astype(bool)
            out[idx_ch] = out[idx_ch] & res & limarr

    return Array(nda=out)


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
) -> Array:
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

        res = get_data_at_channel(
            f"ch{ch}",
            ids,
            idx,
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

    return Array(nda=out)


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
) -> VectorOfVectors:
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
    out = ch_comp.to_aoesa().view_as("np")

    chns = np.unique(out[~np.isnan(out)]).astype(int)

    type_name = None
    for ch in chns:
        res = get_data_at_channel(
            f"ch{ch}",
            ids,
            idx,
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
    return out


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
) -> ArrayOfEqualSizedArrays:
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
        res = get_data_at_channel(
            ch,
            ids,
            idx,
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
        limarr = get_mask_from_query(qry, len(res), ch, ids, idx, f_hit, f_dsp)

        # append to out according to mode == vov
        out[:, i][limarr] = res[limarr]

        i += 1

    return ArrayOfEqualSizedArrays(nda=out)


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
) -> VectorOfVectors:
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
    ).view_as("np")

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
        ).view_as("np")
        if "ascend_by" == md:
            out[np.arange(len(out))[:, None], np.argsort(s_val)]

        elif "descend_by" == md:
            out[np.arange(len(out))[:, None], np.argsort(-s_val)]
        else:
            raise ValueError(
                "sorter values can only have 'ascend_by' or 'descend_by' prefixes"
            )

    out = VectorOfVectors(
        flattened_data=out.flatten()[~np.isnan(out.flatten())].astype(type(defv)),
        cumulative_length=np.cumsum(np.count_nonzero(~np.isnan(out), axis=1)),
    )

    return out


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

    nrows = store.read_n_rows(f"{tcm_group}/cumulative_length", f_tcm)

    table = Table(size=nrows)

    for k, v in tbl_cfg["operations"].items():
        log.debug("Processing field" + k)

        # if mode not defined in operation, it can only be an operation on the evt level.
        if "aggregation_mode" not in v.keys():
            var = {}
            if "parameters" in v.keys():
                var = var | v["parameters"]
            res = table.eval(v["expression"].replace("evt.", ""), var)
            table.add_field(k, res)

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
            if "initial" in v.keys():
                defaultv = v["initial"]
                if isinstance(defaultv, str) and (
                    defaultv in ["np.nan", "np.inf", "-np.inf"]
                ):
                    defaultv = eval(defaultv)
            if "sort" in v.keys():
                srter = v["sort"]

            obj = evaluate_expression(
                f_tcm,
                f_hit,
                f_dsp,
                chns_e,
                chns_rm,
                v["aggregation_mode"],
                v["expression"],
                nrows,
                table,
                pars,
                qry,
                defaultv,
                srter,
            )

            table.add_field(k, obj)

    # write output fields into f_evt
    if "outputs" in tbl_cfg.keys():
        if len(tbl_cfg["outputs"]) < 1:
            log.warning("No output fields specified, no file will be written.")
        else:
            clms_to_remove = [e for e in table.keys() if e not in tbl_cfg["outputs"]]
            for fld in clms_to_remove:
                table.remove_field(fld, True)
            store.write(obj=table, name=group, lh5_file=f_evt, wo_mode=wo_mode)
    else:
        log.warning("No output fields specified, no file will be written.")

    key = re.search(r"\d{8}T\d{6}Z", f_hit).group(0)
    log.info(
        f"Applied {len(tbl_cfg['operations'])} operations to key {key} and saved {len(tbl_cfg['outputs'])} evt fields across {len(chns)} channel groups"
    )
