"""
This module implements routines to build the evt tier.
"""

from __future__ import annotations

import json
import logging
import os

import awkward as ak
import h5py
import lgdo.lh5_store as store
import numpy as np
import pandas as pd
from lgdo import Array, ArrayOfEqualSizedArrays, VectorOfVectors

log = logging.getLogger(__name__)


def vov_to_ak(vov: VectorOfVectors) -> ak.Array:
    """
    Temporary function to convert VectorOfVectors to awkward arrays. This function will be removed soon.

    Parameters
    ----------
    vov
       VectorOfVectors to be converted.
    """
    flattened_data = vov.flattened_data
    cumulative_length = vov.cumulative_length
    if isinstance(flattened_data, Array):
        flattened_data = flattened_data.nda
    if isinstance(cumulative_length, Array):
        cumulative_length = cumulative_length.nda

    offsets = np.empty(len(cumulative_length) + 1, dtype=cumulative_length.dtype)
    offsets[1:] = cumulative_length
    offsets[0] = 0

    layout = ak.contents.ListOffsetArray(
        offsets=ak.index.Index(offsets), content=ak.contents.NumpyArray(flattened_data)
    )
    return ak.Array(layout)


def vov_to_aoesa(
    vov: VectorOfVectors, missing_value=np.nan, length: int = None
) -> ArrayOfEqualSizedArrays:
    """
    Temporary function to convert VectorOfVectors to ArrayOfEqualSizedArrays. This function will be removed soon.

    Parameters
    ----------
    vov
       VectorOfVectors to be converted.
    missing_value
       missing value to be inserted. Determines the datatype of the output ArrayOfEqualSizedArrays
    length
       length of each row in the ArrayOfEqualSizedArrays. If the row in VectorOfVectors is shorter than length, the row gets padded with missing_value. If the row in VectorOfVectors is longer than length, the row gets clipped.
    """
    arr = vov_to_ak(vov)
    if length is not None:
        max_len = length
    else:
        max_len = int(ak.max(ak.count(arr, axis=-1)))
    return ArrayOfEqualSizedArrays(
        nda=ak.fill_none(ak.pad_none(arr, max_len, clip=True), missing_value)
        .to_numpy(allow_missing=False)
        .astype(type(missing_value)),
        attrs=vov.getattrs(),
    )


def build_skm(
    f_evt: str | list,
    f_skm: str,
    skm_conf: dict | str,
    wo_mode="w",
    group: str = "/evt/",
    skim_format: str = "parquet",
):
    """
    Builds a skimmed file from a (set) of evt tier file(s).

    Parameters
    ----------
    f_evt
        list/path of evt file(s)
    f_skm
        name of the skm output file
    skm_conf
        name of JSON file or dict defining skm fields. multiplicity defines upto which row length VectorOfVector fields should be kept. Skimmed fields are forwarded from the evt tier and clipped/padded according to missing_value if needed. Global fields define an operation to reduce the dimension of VectorOfVector event fields.
        For example:

        .. code-block::json

            {
                "multiplicity": 2,
                "index_field": "timestamp",
                "skimmed_fields": {
                    "timestamp":{
                        "evt_field": "timestamp"
                    },
                    "is_muon_rejected":{
                        "evt_field": "is_muon_rejected"
                    },
                    "multiplicity":{
                        "evt_field": "multiplicity"
                    },
                    "energy":{
                        "evt_field": "energy",
                        "missing_value": "np.nan"
                    },
                    "energy_id":{
                        "evt_field": "energy_id",
                        "missing_value": 0
                    },
                    "global_fields":{
                        "energy_sum":{
                            "aggregation_mode": "sum",
                            "evt_field": "energy"
                        },
                        "is_all_physical":{
                            "aggregation_mode": "all",
                            "evt_field": "is_physical"
                        },
                    }
                }
            }

    wo_mode
        writing mode.
        - ``write_safe`` or ``w``: only proceed with writing if the file does not already exis.
        - ``append`` or ``a``: append  to file.
        - ``overwrite`` or ``o``: replaces existing file.
    group
        lh5 root group name of the evt tier
    skim_format
        data format of the skimmed output (hdf or parquet)
    """

    log = logging.getLogger(__name__)
    log.info("Starting skimming")
    log.debug(f"I am skimning {len(f_evt) if isinstance(f_evt,list) else 1} files")
    tbl_cfg = skm_conf
    if not isinstance(tbl_cfg, (str, dict)):
        raise TypeError()
    if isinstance(tbl_cfg, str):
        with open(tbl_cfg) as f:
            tbl_cfg = json.load(f)

    flds, flds_vov, flds_arr, multi = None, None, None, None
    if "skimmed_fields" in tbl_cfg.keys():
        flds = tbl_cfg["skimmed_fields"].keys()
        evt_flds = [(e, tbl_cfg["skimmed_fields"][e]["evt_field"]) for e in flds]
        f = h5py.File(f_evt[0] if isinstance(f_evt, list) else f_evt, "r")
        flds_vov = [
            x
            for x in evt_flds
            if x[1]
            in [
                e.split("/")[-1]
                for e in store.ls(f_evt[0] if isinstance(f_evt, list) else f_evt, group)
                if "array<1>{array<1>{" in f[e].attrs.get("datatype")
            ]
        ]
        flds_arr = [
            x
            for x in evt_flds
            if x not in flds_vov
            and x[1]
            in [
                e.split("/")[-1]
                for e in store.ls(f_evt[0] if isinstance(f_evt, list) else f_evt, group)
            ]
        ]

    gflds = None
    if "global_fields" in tbl_cfg.keys():
        gflds = list(tbl_cfg["global_fields"].keys())

    if flds is None and gflds is None:
        return

    # Check if multiplicity is given, if vector like fields are skimmed
    if (
        isinstance(flds_vov, list)
        and len(flds_vov) > 0
        and "multiplicity" not in tbl_cfg.keys()
    ):
        raise ValueError("If skiime fields are passed, multiplicity must be given")

    elif "multiplicity" in tbl_cfg.keys():
        multi = tbl_cfg["multiplicity"]

    # init pandas df
    df = pd.DataFrame()

    # add array like fields
    if isinstance(flds_arr, list):
        log.debug("Crunching array-like fields")
        df = df.join(
            store.load_dfs(f_evt, [x[1] for x in flds_arr], group).rename(
                columns={y: x for x, y in flds_arr}
            ),
            how="outer",
        )

    # take care of vector like fields
    if isinstance(flds_vov, list):
        log.debug("Processing VoV-like fields")
        lstore = store.LH5Store()
        for fld in flds_vov:
            if "missing_value" not in tbl_cfg["skimmed_fields"][fld[0]].keys():
                raise ValueError(
                    f"({fld[0]}) is a VectorOfVector field and no missing_value is specified"
                )
            vls, _ = lstore.read_object(group + fld[1], f_evt)
            mv = tbl_cfg["skimmed_fields"][fld[0]]["missing_value"]
            if mv in ["np.inf", "-np.inf", "np.nan"]:
                mv = eval(mv)
            out = vov_to_aoesa(vls, missing_value=mv, length=multi).nda
            nms = [fld[0] + f"_{e}" for e in range(multi)]
            df = df.join(pd.DataFrame(data=out, columns=nms), how="outer")

    # ok now build global fields if requested
    if isinstance(gflds, list):
        log.debug("Defining global fields")
        for k in gflds:
            if "aggregation_mode" not in tbl_cfg["global_fields"][k].keys():
                raise ValueError(f"global {k} operation needs aggregation mode")
            if "evt_field" not in tbl_cfg["global_fields"][k].keys():
                raise ValueError(f"global {k} operation needs evt_field")
            mode = tbl_cfg["global_fields"][k]["aggregation_mode"]
            fld = tbl_cfg["global_fields"][k]["evt_field"]

            obj, _ = lstore.read_object(group + fld, f_evt)
            if not isinstance(obj, VectorOfVectors):
                raise ValueError(
                    f"global {k} operation not possible, since {fld} is not an VectorOfVectors"
                )

            obj_ak = vov_to_ak(obj)
            if mode in [
                "sum",
                "prod",
                "nansum",
                "nanprod",
                "any",
                "all",
                "mean",
                "std",
                "var",
            ]:
                df = df.join(
                    pd.DataFrame(
                        data=getattr(ak, mode)(obj_ak, axis=-1).to_numpy(
                            allow_missing=False
                        ),
                        columns=[k],
                    )
                )

            elif mode in ["min", "max"]:
                val = getattr(ak, mode)(obj_ak, axis=-1, mask_identity=True)
                if "missing_value" not in tbl_cfg["global_fields"][k].keys():
                    raise ValueError(
                        f"global {k} {mode} operation needs a missing value assigned"
                    )
                mv = tbl_cfg["global_fields"][k]["missing_value"]
                if mv in ["np.inf", "-np.inf"]:
                    mv = eval(mv)
                val = ak.fill_none(val, mv)
                df = df.join(
                    pd.DataFrame(data=val.to_numpy(allow_missing=False), columns=[k])
                )
            else:
                raise ValueError("aggregation mode not supported")

    # Set an index column if specified
    if "index_field" in tbl_cfg.keys():
        log.debug("Setting index")
        if tbl_cfg["index_field"] in df.keys():
            df = df.set_index(tbl_cfg["index_field"])
        else:
            raise ValueError(
                "index field not found. Needs to be a previously defined skm field"
            )

    # last thing missing is writing it out
    log.debug("saving skm file")
    if skim_format not in ["parquet", "hdf"]:
        raise ValueError("Not supported skim data format. Operations are hdf, parquet")
    if wo_mode in ["w", "write_safe"]:
        if os.path.exists(f_skm):
            raise FileExistsError(f"Write_safe mode: {f_skm} exists.")
        else:
            if "hdf" == skim_format:
                df.to_hdf(f_skm, key="df", mode="w")
            elif "parquet" == skim_format:
                df.to_parquet(f_skm)
    elif wo_mode in ["o", "overwrite"]:
        if "hdf" == skim_format:
            df.to_hdf(f_skm, key="df", mode="w")
        elif "parquet" == skim_format:
            df.to_parquet(f_skm)
    elif wo_mode in ["a", "append"]:
        if "hdf" == skim_format:
            df.to_hdf(f_skm, key="df", mode="a")
        elif "parquet" == skim_format:
            df.to_parquet(f_skm, append=True)
    else:
        raise ValueError(f"wo_mode {wo_mode} not valid.")

    log.info("done")
