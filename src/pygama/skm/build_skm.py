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

from pygama.evt import utils

log = logging.getLogger(__name__)


def build_skm(
    f_evt: str,
    f_hit: str,
    f_dsp: str,
    f_tcm: str,
    skm_conf: dict | str,
    f_skm: str | None = None,
    wo_mode: str = "w",
    skm_group: str = "skm",
    evt_group: str = "evt",
    tcm_group: str = "hardware_tcm_1",
    dsp_group: str = "dsp",
    hit_group: str = "hit",
    tcm_id_table_pattern: str = "ch{}",
) -> None | Table:
    """Builds a skimmed file from a (set) of `evt/hit/dsp` tier file(s).

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
    skm_conf
        name of configuration file or dictionary defining `skm` fields.

        - ``multiplicity`` defines up to which row length
          :class:`.VectorOfVector` fields should be kept.
        - ``postfixes`` list of postfixes must be list of
          ``len(multiplicity)``. If not given, numbers from 0 to
          ``multiplicity -1`` are used
        - ``operations`` are forwarded from lower tiers and clipped/padded
          according to ``missing_value`` if needed. If the forwarded field
          is not an evt tier, ``tcm_idx`` must be passed that specifies the
          value to pick across channels.

        For example:

        .. code-block:: json

           {
             "multiplicity": 2,
             "postfixes":["", "aux"],
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
    f_skm
        name of the `skm` output file. If ``None``, return the output
        class:`.Table` instead of writing to disk.

    wo_mode
        writing mode.

        - ``write_safe`` or ``w``: only proceed with writing if the file does
          not already exists.
        - ``append`` or ``a``: append  to file.
        - ``overwrite`` or ``o``: replaces existing file.

    skm_group
        `skm` LH5 root group name.
    evt_group
        `evt` LH5 root group name.
    hit_group
        `hit` LH5 root group name.
    dsp_group
        `dsp` LH5 root group name.
    tcm_group
        `tcm` LH5 root group name.
    tcm_id_table_pattern
        pattern to format `tcm` id values to table name in higher tiers. Must have one
        placeholder which is the `tcm` id.
    """
    f_dict = {evt_group: f_evt, hit_group: f_hit, dsp_group: f_dsp, tcm_group: f_tcm}
    log = logging.getLogger(__name__)
    log.debug(f"I am skimming {len(f_evt) if isinstance(f_evt,list) else 1} files")

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

            fw_fld = tbl_cfg["operations"][op]["forward_field"]

            # load object if from evt tier
            if evt_group in fw_fld.replace(".", "/"):
                obj = store.read(
                    f"/{fw_fld.replace('.','/')}", f_dict[fw_fld.split(".", 1)[0]]
                )[0].view_as("ak")

            # else collect data from lower tier via tcm_idx
            else:
                if "tcm_idx" not in tbl_cfg["operations"][op].keys():
                    raise ValueError(
                        f"{op} is an sub evt level operation. tcm_idx field must be specified"
                    )
                tcm_idx_fld = tbl_cfg["operations"][op]["tcm_idx"]
                tcm_idx = store.read(
                    f"/{tcm_idx_fld.replace('.','/')}",
                    f_dict[tcm_idx_fld.split(".")[0]],
                )[0].view_as("ak")[:, :multi]

                obj = ak.Array([[] for x in range(len(tcm_idx))])

                # load TCM data to define an event
                ids = store.read(f"/{tcm_group}/array_id", f_tcm)[0].view_as("ak")
                ids = ak.unflatten(ids[ak.flatten(tcm_idx)], ak.count(tcm_idx, axis=-1))

                idx = store.read(f"/{tcm_group}/array_idx", f_tcm)[0].view_as("ak")
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

                        if (
                            f"{utils.get_table_name_by_pattern(tcm_id_table_pattern,ch)}/{fw_fld.replace('.','/')}"
                            not in lh5.ls(
                                f_dict[[key for key in f_dict if key in fw_fld][0]],
                                f"ch{ch}/{fw_fld.rsplit('.',1)[0]}/",
                            )
                        ):
                            och = Array(nda=np.full(len(fl_idx), miss_val))
                        else:
                            och, _ = store.read(
                                f"{utils.get_table_name_by_pattern(tcm_id_table_pattern,ch)}/{fw_fld.replace('.','/')}",
                                f_dict[[key for key in f_dict if key in fw_fld][0]],
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

    if not f_skm:
        return table

    # last thing missing is writing it out
    if wo_mode not in ["w", "write_safe", "o", "overwrite", "a", "append"]:
        raise ValueError(f"wo_mode {wo_mode} not valid.")

    log.debug("saving skm file")
    if (wo_mode in ["w", "write_safe"]) and os.path.exists(f_skm):
        raise FileExistsError(f"Write_safe mode: {f_skm} exists.")

    wo = wo_mode if wo_mode not in ["o", "overwrite"] else "of"
    store.write(obj=table, name=f"/{skm_group}/", lh5_file=f_skm, wo_mode=wo)
