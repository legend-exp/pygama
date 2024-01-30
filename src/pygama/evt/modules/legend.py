"""
Module provides LEGEND internal functions
"""
from importlib import import_module

from lgdo.lh5 import utils


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
