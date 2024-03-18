"""
This module provides utilities to build the `evt` tier.
"""
from __future__ import annotations

import re
from collections import namedtuple

import awkward as ak
import numpy as np
from lgdo import lh5
from numpy.typing import NDArray

H5DataLoc = namedtuple("H5DataLoc", ("file", "group"), defaults=2 * (None,))
TierData = namedtuple(
    "TierData", ("raw", "tcm", "dsp", "hit", "evt"), defaults=5 * (None,)
)

TCMData = namedtuple("TCMData", ("id", "idx", "cumulative_length"))


def make_files_config(data):
    if not isinstance(data, TierData):
        return TierData(
            *[
                H5DataLoc(*data[tier]) if tier in data else H5DataLoc()
                for tier in TierData._fields
            ]
        )

    return data


def get_tcm_id_by_pattern(chname_fmt: str, ch: str) -> int:
    pre = chname_fmt.split("{")[0]
    post = chname_fmt.split("}")[1]
    return int(ch.strip(pre).strip(post))


def get_table_name_by_pattern(chname_fmt: str, ch_id: int) -> str:
    # check chname_fmt validity
    pattern_check = re.findall(r"{([^}]*?)}", chname_fmt)[0]
    if pattern_check == "" or ":" == pattern_check[0]:
        return chname_fmt.format(ch_id)
    else:
        raise NotImplementedError(
            "Only empty placeholders with format specifications are currently implemented"
        )


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


def find_parameters(
    files_cfg,
    ch: str,
    idx_ch: NDArray,
    exprl: list,
) -> dict:
    """Finds and returns parameters from `hit` and `dsp` tiers.

    Parameters
    ----------
    files_cfg
        input and output LH5 files_cfg with HDF5 groups where tables are found.
    ch
       "rawid" in the tiers.
    idx_ch
       index array of entries to be read from files_cfg.
    exprl
       list of tuples ``(tier, field)`` to be found in the `hit/dsp` tiers.
    """
    f = make_files_config(files_cfg)

    # find fields in either dsp, hit
    dsp_flds = [e[1] for e in exprl if e[0] == f.dsp.group]
    hit_flds = [e[1] for e in exprl if e[0] == f.hit.group]

    hit_dict, dsp_dict = {}, {}

    if len(hit_flds) > 0:
        hit_ak = lh5.read_as(
            f"{ch.replace('/','')}/{f.hit.group}/",
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
            f"{ch.replace('/','')}/{f.dsp.group}/",
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
    files_cfg,
    ch: str,
    tcm: TCMData,
    expr: str,
    exprl: list,
    var_ph: dict,
    is_evaluated: bool,
    default_value,
    chname_fmt: str = "ch{}",
) -> np.ndarray:
    """Evaluates an expression and returns the result.

    Parameters
    ----------
    files_cfg
        input and output LH5 files_cfg with HDF5 groups where tables are found.
    ch
       "rawid" of channel to be evaluated.
    tcm
        TCM data arrays in an object that can be accessed by attribute.
    expr
       expression to be evaluated.
    exprl
       list of parameter-tuples ``(root_group, field)`` found in the expression.
    var_ph
       dict of additional parameters that are not channel dependent.
    is_evaluated
       if false, the expression does not get evaluated but an array of default
       values is returned.
    default_value
       default value.
    chname_fmt
        Pattern to format tcm id values to table name in higher tiers. Must have one
        placeholder which is the tcm id.
    """
    f = make_files_config(files_cfg)

    # get index list for this channel to be loaded
    idx_ch = tcm.idx[tcm.id == get_tcm_id_by_pattern(chname_fmt, ch)]
    outsize = len(idx_ch)

    if not is_evaluated:
        res = np.full(outsize, default_value, dtype=type(default_value))
    elif "tcm.array_id" == expr:
        res = np.full(outsize, get_tcm_id_by_pattern(chname_fmt, ch), dtype=int)
    elif "tcm.index" == expr:
        res = np.where(tcm.id == get_tcm_id_by_pattern(chname_fmt, ch))[0]
    else:
        var = find_parameters(
            files_cfg=files_cfg,
            ch=ch,
            idx_ch=idx_ch,
            exprl=exprl,
        )

        if var_ph is not None:
            var = var | var_ph

        # evaluate expression
        # move tier+dots in expression to underscores (e.g. evt.foo -> evt_foo)
        res = eval(
            expr.replace(f"{f.dsp.group}.", f"{f.dsp.group}_")
            .replace(f"{f.hit.group}.", f"{f.hit.group}_")
            .replace(f"{f.evt.group}.", ""),
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
                f"expression '{expr}' must return 1D array. If you are using "
                "VectorOfVectors or ArrayOfEqualSizedArrays, use awkward "
                "reduction functions to reduce the dimension"
            )

    return res


def get_mask_from_query(
    files_cfg,
    query: str | NDArray,
    length: int,
    ch: str,
    idx_ch: NDArray,
) -> np.ndarray:
    """Evaluates a query expression and returns a mask accordingly.

    Parameters
    ----------
    files_cfg
        input and output LH5 files_cfg with HDF5 groups where tables are found.
    query
       query expression.
    length
       length of the return mask.
    ch
       "rawid" of channel to be evaluated.
    idx_ch
       channel indices to be read.
    """
    f = make_files_config(files_cfg)

    # get sub evt based query condition if needed
    if isinstance(query, str):
        query_lst = re.findall(r"(hit|dsp).([a-zA-Z_$][\w$]*)", query)
        query_var = find_parameters(
            files_cfg=files_cfg,
            ch=ch,
            idx_ch=idx_ch,
            exprl=query_lst,
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
