"""
This module provides utilities to build the `evt` tier.
"""

from __future__ import annotations

import re

import awkward as ak
import numpy as np
from lgdo.lh5 import LH5Store
from numpy.typing import NDArray


def get_tcm_id_by_pattern(tcm_id_table_pattern: str, ch: str) -> int:
    pre = tcm_id_table_pattern.split("{")[0]
    post = tcm_id_table_pattern.split("}")[1]
    return int(ch.strip(pre).strip(post))


def get_table_name_by_pattern(tcm_id_table_pattern: str, ch_id: int) -> str:
    # check tcm_id_table_pattern validity
    pattern_check = re.findall(r"{([^}]*?)}", tcm_id_table_pattern)[0]
    if pattern_check == "" or ":" == pattern_check[0]:
        return tcm_id_table_pattern.format(ch_id)
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
    f_hit: str,
    f_dsp: str,
    ch: str,
    idx_ch: NDArray,
    exprl: list,
    hit_group: str = "hit",
    dsp_group: str = "dsp",
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
    dsp_group
        LH5 root group in dsp file.
    hit_group
        LH5 root group in hit file.
    """

    # find fields in either dsp, hit
    dsp_flds = [e[1] for e in exprl if e[0] == dsp_group]
    hit_flds = [e[1] for e in exprl if e[0] == hit_group]

    store = LH5Store()
    hit_dict, dsp_dict = {}, {}
    if len(hit_flds) > 0:
        hit_ak = store.read(
            f"{ch.replace('/','')}/{hit_group}/", f_hit, field_mask=hit_flds, idx=idx_ch
        )[0].view_as("ak")
        hit_dict = dict(
            zip([f"{hit_group}_" + e for e in ak.fields(hit_ak)], ak.unzip(hit_ak))
        )
    if len(dsp_flds) > 0:
        dsp_ak = store.read(
            f"{ch.replace('/','')}/{dsp_group}/", f_dsp, field_mask=dsp_flds, idx=idx_ch
        )[0].view_as("ak")
        dsp_dict = dict(
            zip([f"{dsp_group}_" + e for e in ak.fields(dsp_ak)], ak.unzip(dsp_ak))
        )

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
    defv,
    tcm_id_table_pattern: str = "ch{}",
    evt_group: str = "evt",
    hit_group: str = "hit",
    dsp_group: str = "dsp",
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
    defv
       default value.
    tcm_id_table_pattern
        Pattern to format tcm id values to table name in higher tiers. Must have one
        placeholder which is the tcm id.
    dsp_group
        LH5 root group in dsp file.
    hit_group
        LH5 root group in hit file.
    evt_group
        LH5 root group in evt file.
    """

    # get index list for this channel to be loaded
    idx_ch = idx[ids == get_tcm_id_by_pattern(tcm_id_table_pattern, ch)]
    outsize = len(idx_ch)

    if not is_evaluated:
        res = np.full(outsize, defv, dtype=type(defv))
    elif "tcm.array_id" == expr:
        res = np.full(
            outsize, get_tcm_id_by_pattern(tcm_id_table_pattern, ch), dtype=int
        )
    elif "tcm.index" == expr:
        res = np.where(ids == get_tcm_id_by_pattern(tcm_id_table_pattern, ch))[0]
    else:
        var = find_parameters(
            f_hit=f_hit,
            f_dsp=f_dsp,
            ch=ch,
            idx_ch=idx_ch,
            exprl=exprl,
            hit_group=hit_group,
            dsp_group=dsp_group,
        )

        if var_ph is not None:
            var = var | var_ph

        # evaluate expression
        # move tier+dots in expression to underscores (e.g. evt.foo -> evt_foo)
        res = eval(
            expr.replace(f"{dsp_group}.", f"{dsp_group}_")
            .replace(f"{hit_group}.", f"{hit_group}_")
            .replace(f"{evt_group}.", ""),
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
    idx_ch: NDArray,
    f_hit: str,
    f_dsp: str,
    hit_group: str = "hit",
    dsp_group: str = "dsp",
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
       channel indices to be read.
    f_hit
       path to `hit` tier file.
    f_dsp
       path to `dsp` tier file.
    hit_group
        LH5 root group in hit file.
    dsp_group
        LH5 root group in dsp file.
    """

    # get sub evt based query condition if needed
    if isinstance(qry, str):
        qry_lst = re.findall(r"(hit|dsp).([a-zA-Z_$][\w$]*)", qry)
        qry_var = find_parameters(
            f_hit=f_hit,
            f_dsp=f_dsp,
            ch=ch,
            idx_ch=idx_ch,
            exprl=qry_lst,
            hit_group=hit_group,
            dsp_group=dsp_group,
        )
        limarr = eval(
            qry.replace(f"{dsp_group}.", f"{dsp_group}_").replace(
                f"{hit_group}.", f"{hit_group}_"
            ),
            qry_var,
        )

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
