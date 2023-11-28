"""
Module for importing channel lists from LEGEND meta data
"""
from importlib import import_module


def legend_meta(params: dict) -> list:
    # only import legend meta data when needed.
    # LEGEND collaborators can use the meta keyword
    # While for users w/o access to the LEGEND meta data this is still working
    lm = import_module("legendmeta")
    lmeta = lm.LegendMetadata(path=params["meta_path"])
    chmap = lmeta.channelmap(params["time_key"])
    tmp = [
        f"ch{e}"
        for e in chmap.map("daq.rawid")
        if chmap.map("daq.rawid")[e]["system"] == params["system"]
    ]
    if "usability" not in params.keys():
        return tmp
    else:
        return [
            e
            for e in tmp
            if chmap.map("daq.rawid")[int(e[2:])]["analysis"]["usability"]
            == params["usability"]
        ]
