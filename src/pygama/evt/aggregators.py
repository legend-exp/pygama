"""
This module provides aggregators to build the `evt` tier.
"""

from __future__ import annotations

import awkward as ak
import numpy as np
import pandas as pd
from lgdo import lh5, types

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
    channel_mapping=None,
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
    if not isinstance(datainfo, utils.DataInfo):
        datainfo = utils.make_files_config(datainfo)

    df = None

    for ch in channels:
        table_id = utils.get_tcm_id_by_pattern(datainfo.hit.table_fmt, ch)
        if table_id is None:
            continue

        # get index list for this channel to be loaded
        chan_tcm_indexs = ak.flatten(tcm.table_key) == table_id
        idx_ch = ak.flatten(tcm.row_in_table)[chan_tcm_indexs].to_numpy()

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

            if df is None:
                # define dimension of output array
                out = utils.make_numpy_full(n_rows, default_value, res.dtype)
                df = pd.DataFrame({"sort_field": np.zeros(len(out)), "res": out})

            # get mask from query
            limarr = utils.get_mask_from_query(
                datainfo=datainfo,
                query=query,
                length=len(res),
                ch=ch,
                idx_ch=idx_ch,
            )

            # find if sorter is in hit or dsp
            sort_field = lh5.read_as(
                f"{ch}/{sorter[0]}/{sorter[1]}",
                (
                    datainfo.hit.file
                    if f"{datainfo.hit.group}" == sorter[0]
                    else datainfo.dsp.file
                ),
                idx=idx_ch,
                library="np",
            )

            if sort_field.ndim > 1:
                raise ValueError(f"sorter '{sorter[0]}/{sorter[1]}' must be a 1D array")

            ch_df = pd.DataFrame({"sort_field": sort_field, "res": res})

            evt_ids_ch = np.repeat(
                np.arange(0, len(tcm.table_key)),
                ak.sum(tcm.table_key == table_id, axis=1),
            )

            if is_first:
                if ch == channels[0]:
                    df["sort_field"] = np.inf
                ids = (
                    ch_df.sort_field.to_numpy() < df.sort_field[evt_ids_ch].to_numpy()
                ) & (limarr)
            else:
                ids = (
                    ch_df.sort_field.to_numpy() > df.sort_field[evt_ids_ch].to_numpy()
                ) & (limarr)

            df.loc[evt_ids_ch[ids], list(df.columns)] = ch_df.loc[ids, list(df.columns)]

    return types.Array(nda=df.res.to_numpy())


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
    channel_mapping=None,
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
    if not isinstance(datainfo, utils.DataInfo):
        datainfo = utils.make_files_config(datainfo)
    out = None

    for ch in channels:
        table_id = utils.get_tcm_id_by_pattern(datainfo.hit.table_fmt, ch)
        if table_id is None:
            continue

        # get index list for this channel to be loaded
        chan_tcm_indexs = ak.flatten(tcm.table_key) == table_id
        idx_ch = ak.flatten(tcm.row_in_table)[chan_tcm_indexs].to_numpy()

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

            # get mask from query
            limarr = utils.get_mask_from_query(
                datainfo=datainfo,
                query=query,
                length=len(res),
                ch=ch,
                idx_ch=idx_ch,
            )

            evt_ids_ch = np.repeat(
                np.arange(0, len(tcm.table_key)),
                ak.sum(tcm.table_key == table_id, axis=1),
            )

            # switch through modes
            if mode == "sum":
                if res.dtype == bool:
                    res = res.astype(int)
                if out.dtype == bool:
                    out = out.astype(int)
                out[evt_ids_ch[limarr]] += res[limarr]
            else:
                if res.dtype != bool:
                    res = res.astype(bool)

                if mode == "any":
                    out[evt_ids_ch] |= res & limarr

                if mode == "all":
                    out[evt_ids_ch] &= res & limarr

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
    channel_mapping=None,
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
    if not isinstance(datainfo, utils.DataInfo):
        datainfo = utils.make_files_config(datainfo)
    table_id_fmt = datainfo.hit.table_fmt

    out = None

    for table_id in np.unique(ch_comp.nda.astype(int)):
        table_name = utils.get_table_name_by_pattern(table_id_fmt, table_id)
        # skip default value
        if table_name not in lh5.ls(datainfo.hit.file):
            continue

        # get index list for this channel to be loaded
        chan_tcm_indexs = ak.flatten(tcm.table_key) == table_id
        idx_ch = ak.flatten(tcm.row_in_table)[chan_tcm_indexs].to_numpy()

        evt_ids_ch = np.repeat(
            np.arange(0, len(tcm.table_key)), ak.sum(tcm.table_key == table_id, axis=1)
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

        out[evt_ids_ch] = np.where(
            table_id == ch_comp.nda[idx_ch], res, out[evt_ids_ch]
        )

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
    channel_mapping=None,
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
    if not isinstance(datainfo, utils.DataInfo):
        datainfo = utils.make_files_config(datainfo)

    ch_comp_channels = np.unique(ch_comp.flattened_data.nda).astype(int)

    out = np.full(
        len(ch_comp.flattened_data.nda), default_value, dtype=type(default_value)
    )

    type_name = None
    for table_id in ch_comp_channels:
        table_name = utils.get_table_name_by_pattern(datainfo.hit.table_fmt, table_id)
        evt_ids_ch = np.repeat(
            np.arange(0, len(tcm.table_key)), ak.sum(tcm.table_key == table_id, axis=1)
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
            new_evt_ids_ch = np.repeat(
                np.arange(0, len(ch_comp)),
                ak.sum(ch_comp.view_as("ak") == table_id, axis=1),
            )
            matches = np.isin(evt_ids_ch, new_evt_ids_ch)
            out[ch_comp.flattened_data.nda == table_id] = res[matches]

        else:
            length = len(np.where(ch_comp.flattened_data.nda == table_id)[0])
            res = np.full(length, default_value)
            out[ch_comp.flattened_data.nda == table_id] = res

        if table_id == ch_comp_channels[0]:
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
    channel_mapping=None,
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
    if not isinstance(datainfo, utils.DataInfo):
        datainfo = utils.make_files_config(datainfo)

    # define dimension of output array
    dtype = None
    out = None

    for i, ch in enumerate(channels):
        table_id = utils.get_tcm_id_by_pattern(datainfo.hit.table_fmt, ch)
        if table_id is None:
            continue

        # get index list for this channel to be loaded
        chan_tcm_indexs = ak.flatten(tcm.table_key) == table_id
        idx_ch = ak.flatten(tcm.row_in_table)[chan_tcm_indexs].to_numpy()

        evt_ids_ch = np.repeat(
            np.arange(0, len(tcm.table_key)), ak.sum(tcm.table_key == table_id, axis=1)
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
    channel_mapping=None,
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
