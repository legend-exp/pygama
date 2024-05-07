"""
This module provides aggregators to build the `evt` tier.
"""

from __future__ import annotations

import awkward as ak
import numpy as np
from lgdo import lh5, types
from lgdo.lh5 import LH5Store

from . import utils


def evaluate_to_first_or_last(
    datainfo,
    tcm,
    channels,
    channels_skip,
    expr,
    field_list,
    query,
    n_rows,
    sorter,
    pars_dict=None,
    default_value=np.nan,
    is_first: bool = True,
) -> types.Array:
    """Aggregates across channels by returning the expression of the channel
    with value of `sorter`.

    Parameters
    ----------
    datainfo
        input and output LH5 datainfo with HDF5 groups where tables are found.
    tcm
        TCM data arrays in an object that can be accessed by attribute.
    channels
       list of channels to be aggregated.
    channels_skip
       list of channels to be skipped from evaluation and set to default value.
    expr
       expression string to be evaluated.
    field_list
       list of `dsp/hit/evt` parameter tuples in expression ``(tier, field)``.
    query
       query expression to mask aggregation.
    n_rows
       length of output array.
    sorter
       tuple of field in `hit/dsp/evt` tier to evaluate ``(tier, field)``.
    pars_dict
       dictionary of `evt` and additional parameters and their values.
    default_value
       default value.
    is_first
       defines if sorted by smallest or largest value of `sorter`
    """
    f = utils.make_files_config(datainfo)

    out = None
    outt = None
    store = LH5Store(keep_open=True)

    for ch in channels:
        table_id = utils.get_tcm_id_by_pattern(f.hit.table_fmt, ch)

        # get index list for this channel to be loaded
        idx_ch = tcm.idx[tcm.id == table_id]

        # evaluate at channel
        if ch not in channels_skip:
            res = utils.get_data_at_channel(
                datainfo=datainfo,
                ch=ch,
                tcm=tcm,
                expr=expr,
                field_list=field_list,
                pars_dict=pars_dict,
            )

            if out is None:
                # define dimension of output array
                out = utils.make_numpy_full(n_rows, default_value, res.dtype)
                outt = np.zeros(len(out))
        else:
            res = np.full(len(idx_ch), default_value)

        # get mask from query
        limarr = utils.get_mask_from_query(
            datainfo=datainfo,
            query=query,
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

        evt_ids_ch = np.searchsorted(
            tcm.cumulative_length,
            np.where(tcm.id == table_id)[0],
            "right",
        )

        if is_first:
            if ch == channels[0]:
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

    return types.Array(nda=out)


def evaluate_to_scalar(
    datainfo,
    tcm,
    mode,
    channels,
    channels_skip,
    expr,
    field_list,
    query,
    n_rows,
    pars_dict=None,
    default_value=np.nan,
) -> types.Array:
    """Aggregates by summation across channels.

    Parameters
    ----------
    datainfo
        input and output LH5 datainfo with HDF5 groups where tables are found.
    tcm
        TCM data arrays in an object that can be accessed by attribute.
    mode
       aggregation mode.
    channels
       list of channels to be aggregated.
    channels_skip
       list of channels to be skipped from evaluation and set to default value.
    expr
       expression string to be evaluated.
    field_list
       list of `dsp/hit/evt` parameter tuples in expression ``(tier, field)``.
    query
       query expression to mask aggregation.
    n_rows
       length of output array
    pars_dict
       dictionary of `evt` and additional parameters and their values.
    default_value
       default value.
    """
    f = utils.make_files_config(datainfo)
    out = None

    for ch in channels:
        table_id = utils.get_tcm_id_by_pattern(f.hit.table_fmt, ch)

        # get index list for this channel to be loaded
        idx_ch = tcm.idx[tcm.id == table_id]

        if ch not in channels_skip:
            res = utils.get_data_at_channel(
                datainfo=datainfo,
                ch=ch,
                tcm=tcm,
                expr=expr,
                field_list=field_list,
                pars_dict=pars_dict,
            )

            if out is None:
                # define dimension of output array
                out = utils.make_numpy_full(n_rows, default_value, res.dtype)
        else:
            res = np.full(len(idx_ch), default_value)

        # get mask from query
        limarr = utils.get_mask_from_query(
            datainfo=datainfo,
            query=query,
            length=len(res),
            ch=ch,
            idx_ch=idx_ch,
        )

        evt_ids_ch = np.searchsorted(
            tcm.cumulative_length,
            np.where(tcm.id == table_id)[0],
            side="right",
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

    return types.Array(nda=out)


def evaluate_at_channel(
    datainfo,
    tcm,
    channels,
    channels_skip,
    expr,
    field_list,
    ch_comp,
    pars_dict=None,
    default_value=np.nan,
) -> types.Array:
    """Aggregates by evaluating the expression at a given channel.

    Parameters
    ----------
    datainfo
        input and output LH5 datainfo with HDF5 groups where tables are found.
    tcm
        TCM data arrays in an object that can be accessed by attribute.
    channels
        list of channels to be included for evaluation.
    channels_skip
       list of channels to be skipped from evaluation and set to default value.
    expr
       expression string to be evaluated.
    field_list
       list of `dsp/hit/evt` parameter tuples in expression ``(tier, field)``.
    ch_comp
       array of rawids at which the expression is evaluated.
    pars_dict
       dictionary of `evt` and additional parameters and their values.
    default_value
       default value.
    """
    f = utils.make_files_config(datainfo)
    table_id_fmt = f.hit.table_fmt

    out = None

    for ch in np.unique(ch_comp.nda.astype(int)):
        table_name = utils.get_table_name_by_pattern(table_id_fmt, ch)
        # skip default value
        if table_name not in lh5.ls(f.hit.file):
            continue

        idx_ch = tcm.idx[tcm.id == ch]
        evt_ids_ch = np.searchsorted(
            tcm.cumulative_length, np.where(tcm.id == ch)[0], "right"
        )
        if (table_name in channels) and (table_name not in channels_skip):
            res = utils.get_data_at_channel(
                datainfo=datainfo,
                ch=table_name,
                tcm=tcm,
                expr=expr,
                field_list=field_list,
                pars_dict=pars_dict,
            )
        else:
            res = np.full(len(idx_ch), default_value)

        if out is None:
            out = utils.make_numpy_full(len(ch_comp.nda), default_value, res.dtype)

        out[evt_ids_ch] = np.where(ch == ch_comp.nda[idx_ch], res, out[evt_ids_ch])

    return types.Array(nda=out)


def evaluate_at_channel_vov(
    datainfo,
    tcm,
    expr,
    field_list,
    ch_comp,
    channels,
    channels_skip,
    pars_dict=None,
    default_value=np.nan,
) -> types.VectorOfVectors:
    """Same as :func:`evaluate_at_channel` but evaluates expression at non
    flat channels :class:`.VectorOfVectors`.

    Parameters
    ----------
    datainfo
        input and output LH5 datainfo with HDF5 groups where tables are found.
    tcm
        TCM data arrays in an object that can be accessed by attribute.
    expr
       expression string to be evaluated.
    field_list
       list of `dsp/hit/evt` parameter tuples in expression ``(tier, field)``.
    ch_comp
       array of "rawid"s at which the expression is evaluated.
    channels
       list of channels to be included for evaluation.
    channels_skip
       list of channels to be skipped from evaluation and set to default value.
    pars_dict
       dictionary of `evt` and additional parameters and their values.
    default_value
       default value.
    """
    f = utils.make_files_config(datainfo)

    ch_comp_channels = np.unique(ch_comp.flattened_data.nda).astype(int)

    out = np.full(
        len(ch_comp.flattened_data.nda), default_value, dtype=type(default_value)
    )

    type_name = None
    for ch in ch_comp_channels:
        table_name = utils.get_table_name_by_pattern(f.hit.table_fmt, ch)
        evt_ids_ch = np.searchsorted(
            tcm.cumulative_length, np.where(tcm.id == ch)[0], "right"
        )
        if (table_name in channels) and (table_name not in channels_skip):
            res = utils.get_data_at_channel(
                datainfo=datainfo,
                ch=table_name,
                tcm=tcm,
                expr=expr,
                field_list=field_list,
                pars_dict=pars_dict,
            )
            new_evt_ids_ch = np.searchsorted(
                ch_comp.cumulative_length,
                np.where(ch_comp.flattened_data.nda == ch)[0],
                "right",
            )
            matches = np.isin(evt_ids_ch, new_evt_ids_ch)
            out[ch_comp.flattened_data.nda == ch] = res[matches]

        else:
            length = len(np.where(ch_comp.flattened_data.nda == ch)[0])
            res = np.full(length, default_value)
            out[ch_comp.flattened_data.nda == ch] = res

        if ch == ch_comp_channels[0]:
            out = out.astype(res.dtype)
            type_name = res.dtype

    return types.VectorOfVectors(
        flattened_data=types.Array(out, dtype=type_name),
        cumulative_length=ch_comp.cumulative_length,
    )


def evaluate_to_aoesa(
    datainfo,
    tcm,
    channels,
    channels_skip,
    expr,
    field_list,
    query,
    n_rows,
    pars_dict=None,
    default_value=np.nan,
    missing_value=np.nan,
) -> types.ArrayOfEqualSizedArrays:
    """Aggregates by returning an :class:`.ArrayOfEqualSizedArrays` of evaluated
    expressions of channels that fulfill a query expression.

    Parameters
    ----------
    datainfo
        input and output LH5 datainfo with HDF5 groups where tables are found.
    tcm
        TCM data arrays in an object that can be accessed by attribute.
    channels
       list of channels to be aggregated.
    channels_skip
       list of channels to be skipped from evaluation and set to default value.
    expr
       expression string to be evaluated.
    field_list
       list of `dsp/hit/evt` parameter tuples in expression ``(tier, field)``.
    query
       query expression to mask aggregation.
    n_rows
       length of output :class:`.VectorOfVectors`.
    ch_comp
       array of "rawid"s at which the expression is evaluated.
    pars_dict
       dictionary of `evt` and additional parameters and their values.
    default_value
       default value.
    missing_value
       missing value.
    sorter
       sorts the entries in the vector according to sorter expression.
    """
    f = utils.make_files_config(datainfo)

    # define dimension of output array
    dtype = None
    out = None

    for i, ch in enumerate(channels):
        table_id = utils.get_tcm_id_by_pattern(f.hit.table_fmt, ch)
        idx_ch = tcm.idx[tcm.id == table_id]

        evt_ids_ch = np.searchsorted(
            tcm.cumulative_length,
            np.where(tcm.id == table_id)[0],
            "right",
        )

        if ch not in channels_skip:
            res = utils.get_data_at_channel(
                datainfo=datainfo,
                ch=ch,
                tcm=tcm,
                expr=expr,
                field_list=field_list,
                pars_dict=pars_dict,
            )

            if dtype is None:
                dtype = res.dtype

            if out is None:
                out = utils.make_numpy_full(
                    (n_rows, len(channels)), missing_value, res.dtype
                )
        else:
            res = np.full(len(idx_ch), default_value)

        # get mask from query
        limarr = utils.get_mask_from_query(
            datainfo=datainfo,
            query=query,
            length=len(res),
            ch=ch,
            idx_ch=idx_ch,
        )

        out[evt_ids_ch, i] = np.where(limarr, res, out[evt_ids_ch, i])

    return out, dtype


def evaluate_to_vector(
    datainfo,
    tcm,
    channels,
    channels_skip,
    expr,
    field_list,
    query,
    n_rows,
    pars_dict=None,
    default_value=np.nan,
    sorter=None,
) -> types.VectorOfVectors:
    """Aggregates by returning a :class:`.VectorOfVector` of evaluated
    expressions of channels that fulfill a query expression.

    Parameters
    ----------
    datainfo
        input and output LH5 datainfo with HDF5 groups where tables are found.
    tcm
        TCM data arrays in an object that can be accessed by attribute.
    channels
       list of channels to be aggregated.
    channels_skip
       list of channels to be skipped from evaluation and set to default value.
    expr
       expression string to be evaluated.
    field_list
       list of `dsp/hit/evt` parameter tuples in expression ``(tier, field)``.
    query
       query expression to mask aggregation.
    n_rows
       length of output :class:`.VectorOfVectors`.
    ch_comp
       array of "rawids" at which the expression is evaluated.
    pars_dict
       dictionary of `evt` and additional parameters and their values.
    default_value
       default value.
    sorter
       sorts the entries in the vector according to sorter expression.
       ``ascend_by:<hit|dsp.field>`` results in an vector ordered ascending,
       ``decend_by:<hit|dsp.field>`` sorts descending.
    """
    out, dtype = evaluate_to_aoesa(
        datainfo=datainfo,
        tcm=tcm,
        channels=channels,
        channels_skip=channels_skip,
        expr=expr,
        field_list=field_list,
        query=query,
        n_rows=n_rows,
        pars_dict=pars_dict,
        default_value=default_value,
        missing_value=np.nan,
    )

    # if a sorter is given sort accordingly
    if sorter is not None:
        md, fld = sorter.split(":")
        s_val, _ = evaluate_to_aoesa(
            datainfo=datainfo,
            tcm=tcm,
            channels=channels,
            channels_skip=channels_skip,
            expr=fld,
            field_list=[tuple(fld.split("."))],
            query=None,
            n_rows=n_rows,
            missing_value=np.nan,
        )

        if "ascend_by" == md:
            out = out[np.arange(len(out))[:, None], np.argsort(s_val)]

        elif "descend_by" == md:
            out = out[np.arange(len(out))[:, None], np.argsort(-s_val)]
        else:
            raise ValueError(
                "sorter values can only have 'ascend_by' or 'descend_by' prefixes"
            )

    return types.VectorOfVectors(
        ak.values_astype(ak.drop_none(ak.nan_to_none(ak.Array(out))), dtype)
    )
