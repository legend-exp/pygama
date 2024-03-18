"""
This module provides aggregators to build the `evt` tier.
"""

from __future__ import annotations

import awkward as ak
import numpy as np
from lgdo import Array, ArrayOfEqualSizedArrays, VectorOfVectors, lh5
from lgdo.lh5 import LH5Store
from numpy.typing import NDArray

from . import utils


def evaluate_to_first_or_last(
    files_cfg,
    cumulength: NDArray,
    idx: NDArray,
    ids: NDArray,
    chns: list,
    chns_rm: list,
    expr: str,
    exprl: list,
    qry: str | NDArray,
    nrows: int,
    sorter: tuple,
    var_ph: dict = None,
    default_value: bool | int | float = np.nan,
    is_first: bool = True,
    chname_fmt: str = "ch{}",
) -> Array:
    """Aggregates across channels by returning the expression of the channel
    with value of `sorter`.

    Parameters
    ----------
    files_cfg
        input and output LH5 files_cfg with HDF5 groups where tables are found.
    idx
       `tcm` index array.
    ids
       `tcm` id array.
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
    default_value
       default value.
    is_first
       defines if sorted by smallest or largest value of `sorter`
    chname_fmt
        pattern to format `tcm` id values to table name in higher tiers. Must have one
        placeholder which is the `tcm` id.
    """
    f = utils.make_files_config(files_cfg)

    # define dimension of output array
    out = np.full(nrows, default_value, dtype=type(default_value))
    outt = np.zeros(len(out))

    store = LH5Store()

    for ch in chns:
        # get index list for this channel to be loaded
        idx_ch = idx[ids == utils.get_tcm_id_by_pattern(chname_fmt, ch)]
        evt_ids_ch = np.searchsorted(
            cumulength,
            np.where(ids == utils.get_tcm_id_by_pattern(chname_fmt, ch))[0],
            "right",
        )

        # evaluate at channel
        res = utils.get_data_at_channel(
            files_cfg=files_cfg,
            ch=ch,
            ids=ids,
            idx=idx,
            expr=expr,
            exprl=exprl,
            var_ph=var_ph,
            is_evaluated=ch not in chns_rm,
            default_value=default_value,
            chname_fmt=chname_fmt,
        )

        # get mask from query
        limarr = utils.get_mask_from_query(
            files_cfg=files_cfg,
            qry=qry,
            length=len(res),
            ch=ch,
            idx_ch=idx_ch,
        )

        # find if sorter is in hit or dsp
        t0 = store.read(
            f"{ch}/{sorter[0]}/{sorter[1]}",
            f.hit.file if f"{f.hit.group}" == sorter[0] else f.dsp.file,
            idx=idx_ch,
        )[0].view_as("np")

        if t0.ndim > 1:
            raise ValueError(f"sorter '{sorter[0]}/{sorter[1]}' must be a 1D array")

        if is_first:
            if ch == chns[0]:
                outt[:] = np.inf

            out[evt_ids_ch] = np.where(
                (t0 < outt[evt_ids_ch]) & (limarr), res, out[evt_ids_ch]
            )
            outt[evt_ids_ch] = np.where(
                (t0 < outt[evt_ids_ch]) & (limarr), t0, outt[evt_ids_ch]
            )

        else:
            out[evt_ids_ch] = np.where(
                (t0 > outt[evt_ids_ch]) & (limarr), res, out[evt_ids_ch]
            )
            outt[evt_ids_ch] = np.where(
                (t0 > outt[evt_ids_ch]) & (limarr), t0, outt[evt_ids_ch]
            )

    return Array(nda=out, dtype=type(default_value))


def evaluate_to_scalar(
    files_cfg,
    mode: str,
    cumulength: NDArray,
    idx: NDArray,
    ids: NDArray,
    chns: list,
    chns_rm: list,
    expr: str,
    exprl: list,
    qry: str | NDArray,
    nrows: int,
    var_ph: dict = None,
    default_value: bool | int | float = np.nan,
    chname_fmt: str = "ch{}",
) -> Array:
    """Aggregates by summation across channels.

    Parameters
    ----------
    files_cfg
        input and output LH5 files_cfg with HDF5 groups where tables are found.
    mode
       aggregation mode.
    idx
       `tcm` index array.
    ids
       `tcm` id array.
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
       length of output array
    var_ph
       dictionary of `evt` and additional parameters and their values.
    default_value
       default value.
    chname_fmt
        pattern to format `tcm` id values to table name in higher tiers. Must have one
        placeholder which is the `tcm` id.
    """

    # define dimension of output array
    out = np.full(nrows, default_value, dtype=type(default_value))

    for ch in chns:
        # get index list for this channel to be loaded
        idx_ch = idx[ids == utils.get_tcm_id_by_pattern(chname_fmt, ch)]
        evt_ids_ch = np.searchsorted(
            cumulength,
            np.where(ids == utils.get_tcm_id_by_pattern(chname_fmt, ch))[0],
            "right",
        )

        res = utils.get_data_at_channel(
            files_cfg=files_cfg,
            ch=ch,
            ids=ids,
            idx=idx,
            expr=expr,
            exprl=exprl,
            var_ph=var_ph,
            is_evaluated=ch not in chns_rm,
            default_value=default_value,
            chname_fmt=chname_fmt,
        )

        # get mask from query
        limarr = utils.get_mask_from_query(
            files_cfg=files_cfg,
            qry=qry,
            length=len(res),
            ch=ch,
            idx_ch=idx_ch,
        )

        # switch through modes
        if "sum" == mode:
            if res.dtype == bool:
                res = res.astype(int)
            out[evt_ids_ch] = np.where(limarr, res + out[evt_ids_ch], out[evt_ids_ch])
        if "any" == mode:
            if res.dtype != bool:
                res = res.astype(bool)
            out[evt_ids_ch] = out[evt_ids_ch] | (res & limarr)
        if "all" == mode:
            if res.dtype != bool:
                res = res.astype(bool)
            out[evt_ids_ch] = out[evt_ids_ch] & res & limarr

    return Array(nda=out, dtype=type(default_value))


def evaluate_at_channel(
    files_cfg,
    cumulength: NDArray,
    idx: NDArray,
    ids: NDArray,
    chns_rm: list,
    expr: str,
    exprl: list,
    ch_comp: Array,
    var_ph: dict = None,
    default_value: bool | int | float = np.nan,
    chname_fmt: str = "ch{}",
) -> Array:
    """Aggregates by evaluating the expression at a given channel.

    Parameters
    ----------
    files_cfg
        input and output LH5 files_cfg with HDF5 groups where tables are found.
    idx
       `tcm` index array.
    ids
       `tcm` id array.
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
    default_value
       default value.
    chname_fmt
        pattern to format `tcm` id values to table name in higher tiers. Must have one
        placeholder which is the `tcm` id.
    """
    f = utils.make_files_config(files_cfg)

    out = np.full(len(ch_comp.nda), default_value, dtype=type(default_value))

    for ch in np.unique(ch_comp.nda.astype(int)):
        # skip default value
        if utils.get_table_name_by_pattern(chname_fmt, ch) not in lh5.ls(f.hit.file):
            continue
        idx_ch = idx[ids == ch]
        evt_ids_ch = np.searchsorted(cumulength, np.where(ids == ch)[0], "right")
        res = utils.get_data_at_channel(
            files_cfg=files_cfg,
            ch=utils.get_table_name_by_pattern(chname_fmt, ch),
            ids=ids,
            idx=idx,
            expr=expr,
            exprl=exprl,
            var_ph=var_ph,
            is_evaluated=utils.get_table_name_by_pattern(chname_fmt, ch) not in chns_rm,
            default_value=default_value,
            chname_fmt=chname_fmt,
        )

        out[evt_ids_ch] = np.where(ch == ch_comp.nda[idx_ch], res, out[evt_ids_ch])

    return Array(nda=out, dtype=type(default_value))


def evaluate_at_channel_vov(
    files_cfg,
    cumulength: NDArray,
    idx: NDArray,
    ids: NDArray,
    expr: str,
    exprl: list,
    ch_comp: VectorOfVectors,
    chns_rm: list,
    var_ph: dict = None,
    default_value: bool | int | float = np.nan,
    chname_fmt: str = "ch{}",
) -> VectorOfVectors:
    """Same as :func:`evaluate_at_channel` but evaluates expression at non
    flat channels :class:`.VectorOfVectors`.

    Parameters
    ----------
    files_cfg
        input and output LH5 files_cfg with HDF5 groups where tables are found.
    idx
       `tcm` index array.
    ids
       `tcm` id array.
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
    default_value
       default value.
    chname_fmt
        pattern to format `tcm` id values to table name in higher tiers. Must have one
        placeholder which is the `tcm` id.
    """

    # blow up vov to aoesa
    out = ak.Array([[] for _ in range(len(ch_comp))])

    chns = np.unique(ch_comp.flattened_data.nda).astype(int)
    ch_comp = ch_comp.view_as("ak")

    type_name = None
    for ch in chns:
        evt_ids_ch = np.searchsorted(cumulength, np.where(ids == ch)[0], "right")
        res = utils.get_data_at_channel(
            files_cfg=files_cfg,
            ch=utils.get_table_name_by_pattern(chname_fmt, ch),
            ids=ids,
            idx=idx,
            expr=expr,
            exprl=exprl,
            var_ph=var_ph,
            is_evaluated=utils.get_table_name_by_pattern(chname_fmt, ch) not in chns_rm,
            default_value=default_value,
            chname_fmt=chname_fmt,
        )

        # see in which events the current channel is present
        mask = ak.to_numpy(ak.any(ch_comp == ch, axis=-1), allow_missing=False)
        cv = np.full(len(ch_comp), np.nan)
        cv[evt_ids_ch] = res
        cv[~mask] = np.nan
        cv = ak.drop_none(ak.nan_to_none(ak.Array(cv)[:, None]))

        out = ak.concatenate((out, cv), axis=-1)

        if ch == chns[0]:
            type_name = res.dtype

    return VectorOfVectors(ak.values_astype(out, type_name), dtype=type_name)


def evaluate_to_aoesa(
    files_cfg,
    cumulength: NDArray,
    idx: NDArray,
    ids: NDArray,
    chns: list,
    chns_rm: list,
    expr: str,
    exprl: list,
    qry: str | NDArray,
    nrows: int,
    var_ph: dict = None,
    default_value: bool | int | float = np.nan,
    missv=np.nan,
    chname_fmt: str = "ch{}",
) -> ArrayOfEqualSizedArrays:
    """Aggregates by returning an :class:`.ArrayOfEqualSizedArrays` of evaluated
    expressions of channels that fulfill a query expression.

    Parameters
    ----------
    files_cfg
        input and output LH5 files_cfg with HDF5 groups where tables are found.
    idx
       `tcm` index array.
    ids
       `tcm` id array.
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
    default_value
       default value.
    missv
       missing value.
    sorter
       sorts the entries in the vector according to sorter expression.
    chname_fmt
        pattern to format `tcm` id values to table name in higher tiers. Must have one
        placeholder which is the `tcm` id.
    """
    # define dimension of output array
    out = np.full((nrows, len(chns)), missv)

    i = 0
    for ch in chns:
        idx_ch = idx[ids == utils.get_tcm_id_by_pattern(chname_fmt, ch)]
        evt_ids_ch = np.searchsorted(
            cumulength,
            np.where(ids == utils.get_tcm_id_by_pattern(chname_fmt, ch))[0],
            "right",
        )
        res = utils.get_data_at_channel(
            files_cfg=files_cfg,
            ch=ch,
            ids=ids,
            idx=idx,
            expr=expr,
            exprl=exprl,
            var_ph=var_ph,
            is_evaluated=ch not in chns_rm,
            default_value=default_value,
            chname_fmt=chname_fmt,
        )

        # get mask from query
        limarr = utils.get_mask_from_query(
            files_cfg=files_cfg,
            qry=qry,
            length=len(res),
            ch=ch,
            idx_ch=idx_ch,
        )

        out[evt_ids_ch, i] = np.where(limarr, res, out[evt_ids_ch, i])

        i += 1

    return ArrayOfEqualSizedArrays(nda=out)


def evaluate_to_vector(
    files_cfg,
    cumulength: NDArray,
    idx: NDArray,
    ids: NDArray,
    chns: list,
    chns_rm: list,
    expr: str,
    exprl: list,
    qry: str | NDArray,
    nrows: int,
    var_ph: dict = None,
    default_value: bool | int | float = np.nan,
    sorter: str = None,
    chname_fmt: str = "ch{}",
) -> VectorOfVectors:
    """Aggregates by returning a :class:`.VectorOfVector` of evaluated
    expressions of channels that fulfill a query expression.

    Parameters
    ----------
    files_cfg
        input and output LH5 files_cfg with HDF5 groups where tables are found.
    idx
       `tcm` index array.
    ids
       `tcm` id array.
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
    default_value
       default value.
    sorter
       sorts the entries in the vector according to sorter expression.
       ``ascend_by:<hit|dsp.field>`` results in an vector ordered ascending,
       ``decend_by:<hit|dsp.field>`` sorts descending.
    chname_fmt
        pattern to format `tcm` id values to table name in higher tiers. Must have one
        placeholder which is the `tcm` id.
    """
    out = evaluate_to_aoesa(
        files_cfg=files_cfg,
        cumulength=cumulength,
        idx=idx,
        ids=ids,
        chns=chns,
        chns_rm=chns_rm,
        expr=expr,
        exprl=exprl,
        qry=qry,
        nrows=nrows,
        var_ph=var_ph,
        default_value=default_value,
        missv=np.nan,
        chname_fmt=chname_fmt,
    ).view_as("np")

    # if a sorter is given sort accordingly
    if sorter is not None:
        md, fld = sorter.split(":")
        s_val = evaluate_to_aoesa(
            files_cfg=files_cfg,
            cumulength=cumulength,
            idx=idx,
            ids=ids,
            chns=chns,
            chns_rm=chns_rm,
            expr=fld,
            exprl=[tuple(fld.split("."))],
            qry=None,
            nrows=nrows,
            missv=np.nan,
            chname_fmt=chname_fmt,
        ).view_as("np")
        if "ascend_by" == md:
            out = out[np.arange(len(out))[:, None], np.argsort(s_val)]

        elif "descend_by" == md:
            out = out[np.arange(len(out))[:, None], np.argsort(-s_val)]
        else:
            raise ValueError(
                "sorter values can only have 'ascend_by' or 'descend_by' prefixes"
            )

    return VectorOfVectors(
        ak.values_astype(
            ak.drop_none(ak.nan_to_none(ak.Array(out))), type(default_value)
        ),
        dtype=type(default_value),
    )
