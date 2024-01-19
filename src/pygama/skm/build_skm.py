"""
This module implements routines to build the `skm` tier, consisting of skimmed
data from lower tiers.
"""

from __future__ import annotations

import json
import logging
import os

import awkward as ak
import numpy as np
from lgdo import Array, Table, lh5
from lgdo.lh5 import LH5Store

log = logging.getLogger(__name__)


def build_skm(
    f_evt: str,
    f_hit: str,
    f_dsp: str,
    f_tcm: str,
    f_skm: str,
    skm_conf: dict | str,
    wo_mode="w",
    skim_format: str = "parquet",
    group: str = "/skm/",
) -> None:
    """Builds a skimmed file from a (set) of evt/hit/dsp tier file(s).

    Parameters
    ----------
    f_evt
        path of `evt` file.
    f_hit
        path of `hit` file.
    f_dsp
        path of `dsp` file.
    f_tcm
        path of `tcm` file.
    f_skm
        name of the `skm` output file.
    skm_conf
        name of configuration file or dictionary defining `skm` fields.

        - ``multiplicity`` defines up to which row length
          :class:`.VectorOfVector` fields should be kept.
        - ``postfixes`` list of postfixes must be list of
          ``len(multiplicity)``. If not given, numbers from 0 to
          ``multiplicity -1`` are used
        - ``index_field`` sets the index of the output table. If not given
          the index are set es increasing integers.
        - ``operations`` are forwarded from lower tiers and clipped/padded
          according to ``missing_value`` if needed. If the forwarded field
          is not an evt tier, ``tcm_idx`` must be passed that specifies the
          value to pick across channels.

        For example:

        .. code-block:: json

            {
              "multiplicity": 2,
              "postfixes":["","aux"],
              "index_field": "timestamp",
              "operations": {
                    "timestamp":{
                    "forward_field": "evt.timestamp"
                    },
                    "multiplicity":{
                    "forward_field": "evt.multiplicity"
                    },
                    "energy":{
                    "forward_field": "hit.cuspEmax_ctc_cal",
                    "missing_value": "np.nan",
                    "tcm_idx": "evt.energy_idx"
                    },
                    "energy_id":{
                    "forward_field": "tcm.array_id",
                    "missing_value": 0,
                    "tcm_idx": "evt.energy_idx"
                    }
                }
            }

    wo_mode
        writing mode.

        - ``write_safe`` or ``w``: only proceed with writing if the file does
          not already exists.
        - ``append`` or ``a``: append  to file.
        - ``overwrite`` or ``o``: replaces existing file.

    skim_format
        data format of the skimmed output (``hdf``, ``lh5`` or ``parquet``).
    group
        LH5 root group name (only used if ``skim_format`` is ``lh5``).
    """
    f_dict = {"evt": f_evt, "hit": f_hit, "dsp": f_dsp, "tcm": f_tcm}
    log = logging.getLogger(__name__)
    log.debug(f"I am skimning {len(f_evt) if isinstance(f_evt,list) else 1} files")

    tbl_cfg = skm_conf
    if not isinstance(tbl_cfg, (str, dict)):
        raise TypeError()
    if isinstance(tbl_cfg, str):
        with open(tbl_cfg) as f:
            tbl_cfg = json.load(f)

    # Check if multiplicity is given
    if "multiplicity" not in tbl_cfg.keys():
        raise ValueError("multiplicity field missing")

    multi = int(tbl_cfg["multiplicity"])
    store = LH5Store()
    # df = pd.DataFrame()
    table = Table()
    if "operations" in tbl_cfg.keys():
        for op in tbl_cfg["operations"].keys():
            miss_val = np.nan
            if "missing_value" in tbl_cfg["operations"][op].keys():
                miss_val = tbl_cfg["operations"][op]["missing_value"]
                if isinstance(miss_val, str) and (
                    miss_val in ["np.nan", "np.inf", "-np.inf"]
                ):
                    miss_val = eval(miss_val)

            fw_fld = tbl_cfg["operations"][op]["forward_field"].split(".")
            if fw_fld[0] not in ["evt", "hit", "dsp", "tcm"]:
                raise ValueError(f"{fw_fld[0]} is not a valid tier")

            # load object if from evt tier
            if fw_fld[0] == "evt":
                obj = store.read(f"/{fw_fld[0]}/{fw_fld[1]}", f_dict[fw_fld[0]])[
                    0
                ].view_as("ak")

            # else collect data from lower tier via tcm_idx
            else:
                if "tcm_idx" not in tbl_cfg["operations"][op].keys():
                    raise ValueError(
                        f"{op} is an sub evt level operation. tcm_idx field must be specified"
                    )
                tcm_idx_fld = tbl_cfg["operations"][op]["tcm_idx"].split(".")
                tcm_idx = store.read(
                    f"/{tcm_idx_fld[0]}/{tcm_idx_fld[1]}", f_dict[tcm_idx_fld[0]]
                )[0].view_as("ak")[:, :multi]

                obj = ak.Array([[] for x in range(len(tcm_idx))])

                # load TCM data to define an event
                ids = store.read("hardware_tcm_1/array_id", f_tcm)[0].view_as("ak")
                ids = ak.unflatten(ids[ak.flatten(tcm_idx)], ak.count(tcm_idx, axis=-1))

                idx = store.read("hardware_tcm_1/array_idx", f_tcm)[0].view_as("ak")
                idx = ak.unflatten(idx[ak.flatten(tcm_idx)], ak.count(tcm_idx, axis=-1))

                if "tcm.array_id" == tbl_cfg["operations"][op]["forward_field"]:
                    obj = ids
                elif "tcm.array_idx" == tbl_cfg["operations"][op]["forward_field"]:
                    obj = idx

                else:
                    chns = np.unique(
                        ak.to_numpy(ak.flatten(ids), allow_missing=False)
                    ).astype(int)

                    # Get the data
                    for ch in chns:
                        ch_idx = idx[ids == ch]
                        ct_idx = ak.count(ch_idx, axis=-1)
                        fl_idx = ak.to_numpy(ak.flatten(ch_idx), allow_missing=False)

                        if f"ch{ch}/{fw_fld[0]}/{fw_fld[1]}" not in lh5.ls(
                            f_dict[fw_fld[0]], f"ch{ch}/{fw_fld[0]}/"
                        ):
                            och = Array(nda=np.full(len(fl_idx), miss_val))
                        else:
                            och, _ = store.read(
                                f"ch{ch}/{fw_fld[0]}/{fw_fld[1]}",
                                f_dict[fw_fld[0]],
                                idx=fl_idx,
                            )
                        if not isinstance(och, Array):
                            raise ValueError(
                                f"{type(och)} not supported. Forward only Array fields"
                            )
                        och = och.view_as("ak")
                        och = ak.unflatten(och, ct_idx)
                        obj = ak.concatenate((obj, och), axis=-1)

            # Pad, clip and numpyfy
            if obj.ndim > 1:
                obj = ak.pad_none(obj, multi, clip=True)
            obj = ak.to_numpy(ak.fill_none(obj, miss_val))

            if obj.ndim > 1:
                if "postfixes" in tbl_cfg.keys():
                    nms = [f"{op}{x}" for x in tbl_cfg["postfixes"]]
                else:
                    nms = [f"{op}_{x}" for x in range(multi)]

                for i in range(len(nms)):
                    # add attribute if present
                    ob = Array(nda=obj[:, i])
                    if "lgdo_attrs" in tbl_cfg["operations"][op].keys():
                        ob.attrs |= tbl_cfg["operations"][op]["lgdo_attrs"]
                    table.add_field(nms[i], ob, True)
            else:
                obj = Array(nda=obj)
                if "lgdo_attrs" in tbl_cfg["operations"][op].keys():
                    obj.attrs |= tbl_cfg["operations"][op]["lgdo_attrs"]
                table.add_field(op, obj, True)

    # last thing missing is writing it out
    log.debug("saving skm file")
    if skim_format not in ["parquet", "hdf", "lh5"]:
        raise ValueError(
            "Not supported skim data format. Operations are hdf, lh5, parquet"
        )

    if (wo_mode in ["w", "write_safe"]) and os.path.exists(f_skm):
        raise FileExistsError(f"Write_safe mode: {f_skm} exists.")

    if skim_format in ["hdf", "parquet"]:
        df = table.view_as("pd")
        # Set an index column if specified
        if "index_field" in tbl_cfg.keys():
            log.debug("Setting index")
            if tbl_cfg["index_field"] in df.keys():
                df = df.set_index(tbl_cfg["index_field"])
            else:
                raise ValueError(
                    "index field not found. Needs to be a previously defined skm field"
                )

        if "hdf" == skim_format:
            if wo_mode in ["w", "write_safe", "o", "overwrite"]:
                df.to_hdf(f_skm, key="df", mode="w")
            elif wo_mode in ["a", "append"]:
                df.to_hdf(f_skm, key="df", mode="a")

        elif "parquet" == skim_format:
            if wo_mode in ["w", "write_safe", "o", "overwrite"]:
                df.to_parquet(f_skm)
            elif wo_mode in ["a", "append"]:
                df.to_parquet(f_skm, append=True)

    elif "lh5" == skim_format:
        wo = wo_mode if wo_mode not in ["o", "overwrite"] else "of"
        store.write(obj=table, name=group, lh5_file=f_skm, wo_mode=wo)

    else:
        raise ValueError(f"wo_mode {wo_mode} not valid.")
