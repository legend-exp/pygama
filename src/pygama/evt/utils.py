"""
This module provides utilities to build the `evt` tier.
"""

from __future__ import annotations

import copy
import re
from collections import namedtuple

import awkward as ak
import numpy as np
from lgdo import lh5
from numpy.typing import NDArray

H5DataLoc = namedtuple(
    "H5DataLoc", ("file", "group", "table_fmt"), defaults=3 * (None,)
)

DataInfo = namedtuple(
    "DataInfo", ("raw", "tcm", "dsp", "hit", "evt"), defaults=5 * (None,)
)

TCMData = namedtuple("TCMData", ("id", "idx", "cumulative_length"))


def make_files_config(data: dict):
    if not isinstance(data, DataInfo):
        return DataInfo(
            *[
                H5DataLoc(*data[tier]) if tier in data else H5DataLoc()
                for tier in DataInfo._fields
            ]
        )

    return data


def make_numpy_full(size, fill_value, try_dtype):
    if np.can_cast(fill_value, try_dtype):
        return np.full(size, fill_value, dtype=try_dtype)
    else:
        return np.full(size, fill_value)


def copy_lgdo_attrs(obj):
    attrs = copy.copy(obj.attrs)
    attrs.pop("datatype")
    return attrs


def get_tcm_id_by_pattern(table_id_fmt: str, ch: str) -> int:
    pre = table_id_fmt.split("{")[0]
    post = table_id_fmt.split("}")[1]
    return int(ch.strip(pre).strip(post))


def get_table_name_by_pattern(table_id_fmt: str, ch_id: int) -> str:
    # check table_id_fmt validity
    pattern_check = re.findall(r"{([^}]*?)}", table_id_fmt)[0]
    if pattern_check == "" or ":" == pattern_check[0]:
        return table_id_fmt.format(ch_id)
    else:
        raise NotImplementedError(
            "only empty placeholders {} in format specifications are currently supported"
        )


def find_parameters(
    datainfo,
    ch,
    idx_ch,
    field_list,
) -> dict:
    """Finds and returns parameters from `hit` and `dsp` tiers.

    Parameters
    ----------
    datainfo
        input and output LH5 datainfo with HDF5 groups where tables are found.
    ch
       "rawid" in the tiers.
    idx_ch
       index array of entries to be read from datainfo.
    field_list
       list of tuples ``(tier, field)`` to be found in the `hit/dsp` tiers.
    """
    f = make_files_config(datainfo)

    # find fields in either dsp, hit
    dsp_flds = [e[1] for e in field_list if e[0] == f.dsp.group]
    hit_flds = [e[1] for e in field_list if e[0] == f.hit.group]

    hit_dict, dsp_dict = {}, {}

    if len(hit_flds) > 0:
        hit_ak = lh5.read_as(
            f"{ch.replace('/', '')}/{f.hit.group}/",
            f.hit.file,
            field_mask=hit_flds,
            idx=idx_ch,
            library="ak",
        )

        hit_dict = dict(
            zip([f"{f.hit.group}_" + e for e in ak.fields(hit_ak)], ak.unzip(hit_ak))
        )

    if len(dsp_flds) > 0:
        dsp_ak = lh5.read_as(
            f"{ch.replace('/', '')}/{f.dsp.group}/",
            f.dsp.file,
            field_mask=dsp_flds,
            idx=idx_ch,
            library="ak",
        )

        dsp_dict = dict(
            zip([f"{f.dsp.group}_" + e for e in ak.fields(dsp_ak)], ak.unzip(dsp_ak))
        )

    return hit_dict | dsp_dict


def get_data_at_channel(
    datainfo,
    ch,
    tcm,
    expr,
    field_list,
    pars_dict,
) -> NDArray:
    """Evaluates an expression and returns the result.

    Parameters
    ----------
    datainfo
        input and output LH5 datainfo with HDF5 groups where tables are found.
    ch
       "rawid" of channel to be evaluated.
    tcm
        TCM data arrays in an object that can be accessed by attribute.
    expr
       expression to be evaluated.
    field_list
       list of parameter-tuples ``(root_group, field)`` found in the expression.
    pars_dict
       dict of additional parameters that are not channel dependent.
    is_evaluated
       if false, the expression does not get evaluated but an array of default
       values is returned.
    default_value
       default value.
    """
    f = make_files_config(datainfo)
    table_id = get_tcm_id_by_pattern(f.hit.table_fmt, ch)

    # get index list for this channel to be loaded
    idx_ch = tcm.idx[tcm.id == table_id]
    outsize = len(idx_ch)

    if expr == "tcm.array_id":
        res = np.full(outsize, table_id, dtype=int)
    elif expr == "tcm.array_idx":
        res = idx_ch
    elif expr == "tcm.index":
        res = np.where(tcm.id == table_id)[0]
    else:
        var = find_parameters(
            datainfo=datainfo,
            ch=ch,
            idx_ch=idx_ch,
            field_list=field_list,
        )

        if pars_dict is not None:
            var = var | pars_dict

        # evaluate expression
        # move tier+dots in expression to underscores (e.g. evt.foo -> evt_foo)
        res = eval(
            expr.replace(f"{f.dsp.group}.", f"{f.dsp.group}_")
            .replace(f"{f.hit.group}.", f"{f.hit.group}_")
            .replace(f"{f.evt.group}.", ""),
            var,
        )

        # in case the expression evaluates to a single value blow it up
        if not hasattr(res, "__len__") or isinstance(res, str):
            return np.full(outsize, res)

        # the resulting arrays need to be 1D from the operation,
        # this can only change once we support larger than two dimensional LGDOs
        # ak.to_numpy() raises error if array not regular
        res = ak.to_numpy(res, allow_missing=False)

        # in this method only 1D values are allowed
        if res.ndim > 1:
            raise ValueError(
                f"expression '{expr}' must return 1D array. If you are using "
                "VectorOfVectors or ArrayOfEqualSizedArrays, use awkward "
                "reduction functions to reduce the dimension"
            )

    return res


def get_mask_from_query(
    datainfo,
    query,
    length,
    ch,
    idx_ch,
) -> NDArray:
    """Evaluates a query expression and returns a mask accordingly.

    Parameters
    ----------
    datainfo
        input and output LH5 datainfo with HDF5 groups where tables are found.
    query
       query expression.
    length
       length of the return mask.
    ch
       "rawid" of channel to be evaluated.
    idx_ch
       channel indices to be read.
    """
    f = make_files_config(datainfo)

    # get sub evt based query condition if needed
    if isinstance(query, str):
        query_lst = re.findall(r"(hit|dsp).([a-zA-Z_$][\w$]*)", query)
        query_var = find_parameters(
            datainfo=datainfo,
            ch=ch,
            idx_ch=idx_ch,
            field_list=query_lst,
        )
        limarr = eval(
            query.replace(f"{f.dsp.group}.", f"{f.dsp.group}_").replace(
                f"{f.hit.group}.", f"{f.hit.group}_"
            ),
            query_var,
        )

        # in case the expression evaluates to a single value blow it up
        if (not hasattr(limarr, "__len__")) or (isinstance(limarr, str)):
            return np.full(len(idx_ch), limarr)

        limarr = ak.to_numpy(limarr, allow_missing=False)
        if limarr.ndim > 1:
            raise ValueError(
                f"query '{query}' must return 1D array. If you are using "
                "VectorOfVectors or ArrayOfEqualSizedArrays, use awkward "
                "reduction functions to reduce the dimension"
            )

    # or forward the array
    elif isinstance(query, np.ndarray):
        limarr = query

    # if no condition, it must be true
    else:
        limarr = np.ones(length).astype(bool)

    # explicit cast to bool
    if limarr.dtype != bool:
        limarr = limarr.astype(bool)

    return limarr
