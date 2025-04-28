"""
Module provides LEGEND internal functions
"""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module

import numpy as np
from lgdo import types

from .. import utils


def metadata(params: dict) -> list:
    # only import legend meta data when needed.
    # LEGEND collaborators can use the meta keyword
    # While for users w/o access to the LEGEND meta data this is still working
    lm = import_module("legendmeta")
    lmeta = lm.LegendMetadata(path=utils.expand_path(params["meta_path"]))
    chmap = lmeta.channelmap(params["time_key"])

    tmp = [
        f"ch{e}"
        for e in chmap.map("daq.rawid")
        if chmap.map("daq.rawid")[e]["system"] == params["system"]
    ]

    if "selectors" in params.keys():
        for k in params["selectors"].keys():
            s = ""
            for e in k.split("."):
                s += f"['{e}']"

            tmp = [
                e
                for e in tmp
                if eval("dotter" + s, {"dotter": chmap.map("daq.rawid")[int(e[2:])]})
                == params["selectors"][k]
            ]
    return tmp


def convert_rawid(
    datainfo: utils.DataInfo,
    tcm: utils.TCMData,
    table_names: Sequence[str],
    channel_mapping: dict,
    *,
    rawid_obj: types.VectorOfVectors | types.Array,
):
    """Convert rawid to channel number."""

    if isinstance(rawid_obj, types.VectorOfVectors):
        rawids = rawid_obj.flattened_data.nda
        detector = np.array(
            [
                channel_mapping[datainfo.hit.table_fmt.replace("{}", str(rawid))]
                for rawid in rawids
            ]
        )
        return types.VectorOfVectors(
            flattened_data=detector, cumulative_length=rawid_obj.cumulative_length
        )
    elif isinstance(rawid_obj, types.Array):
        rawids = rawid_obj.nda
        detector = np.array(
            [
                channel_mapping[datainfo.hit.table_fmt.replace("{}", str(rawid))]
                for rawid in rawids
            ]
        )
        return types.Array(detector)
    else:
        raise TypeError("rawid_obj must be a VectorOfVectors or Array")
