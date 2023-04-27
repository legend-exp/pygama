"""Utility functions."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from pygama.lgdo import (
    Array,
    ArrayOfEqualSizedArrays,
    Table,
    VectorOfVectors,
    WaveformTable,
)

log = logging.getLogger(__name__)


def to_datetime(key: str) -> datetime:
    """Convert LEGEND cycle key to :class:`~datetime.datetime`.

    Assumes `key` is formatted as ``YYYYMMDDTHHMMSSZ`` (UTC).
    """
    m = re.match(r"^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z$", key)
    if m is None:
        raise ValueError(f"Could not parse '{key}' as a datetime object")
    else:
        g = [int(el) for el in m.groups()]
        return datetime(*g, tzinfo=timezone.utc)


def to_unixtime(key: str) -> int:
    """Convert LEGEND cycle key to `POSIX timestamp <https://en.wikipedia.org/wiki/Unix_time>`_."""
    return int(to_datetime(key).timestamp())


def inplace_sort(df: pd.DataFrame, by: str) -> None:
    # sort rows according to timestamps
    log.debug(f"sorting database entries according to {by}")
    if by == "timestamp":
        df["_datetime"] = df["timestamp"].apply(to_datetime)
        df.sort_values("_datetime", ignore_index=True, inplace=True)
        df.drop("_datetime", axis=1, inplace=True)
    else:
        df.sort_values(by, ignore_index=True, inplace=True)


def dict_to_table(col_dict: dict, attr_dict: dict):
    for col in col_dict.keys():
        if isinstance(col_dict[col], list):
            if isinstance(col_dict[col][0], (list, np.ndarray, Array)):
                # Convert to VectorOfVectors if there is array-like in a list
                col_dict[col] = VectorOfVectors(
                    listoflists=col_dict[col], attrs=attr_dict[col]
                )
            else:
                # Elements are scalars, convert to Array
                nda = np.array(col_dict[col])
                col_dict[col] = Array(nda=nda, attrs=attr_dict[col])
        elif isinstance(col_dict[col], dict):
            # Dicts are Tables
            col_dict[col] = dict_to_table(
                col_dict=col_dict[col], attr_dict=attr_dict[col]
            )
        else:
            # ndas are Arrays or AOESA
            nda = np.array(col_dict[col])
            if len(nda.shape) == 2:
                dt = attr_dict[col]["datatype"]
                g = re.match(r"\w+<(\d+),(\d+)>{\w+}", dt).groups()
                dims = [int(e) for e in g]
                col_dict[col] = ArrayOfEqualSizedArrays(
                    dims=dims, nda=nda, attrs=attr_dict[col]
                )
            else:
                col_dict[col] = Array(nda=nda, attrs=attr_dict[col])
        attr_dict.pop(col)
    if set(col_dict.keys()) == {"t0", "dt", "values"}:
        return WaveformTable(
            t0=col_dict["t0"],
            dt=col_dict["dt"],
            values=col_dict["values"],
            attrs=attr_dict,
        )
    else:
        return Table(col_dict=col_dict)


def fill_col_dict(
    tier_table: Table,
    col_dict: dict,
    attr_dict: dict,
    tcm_idx: list | pd.RangeIndex,
    table_length: int,
    aoesa_to_vov: bool,
):
    # Put the information from the tier_table (after the columns have been exploded)
    # into col_dict, which will be turned into the final Table
    for col in tier_table.keys():
        if col not in attr_dict.keys():
            attr_dict[col] = tier_table[col].attrs
        else:
            if attr_dict[col] != tier_table[col].attrs:
                if isinstance(tier_table[col], Table):
                    temp_attr = {
                        k: attr_dict[col][k]
                        for k in attr_dict[col].keys() - tier_table[col].keys()
                    }
                    if temp_attr != tier_table[col].attrs:
                        raise ValueError(
                            f"{col} attributes are inconsistent across data"
                        )
                else:
                    raise ValueError(f"{col} attributes are inconsistent across data")
        if isinstance(tier_table[col], ArrayOfEqualSizedArrays):
            # Allocate memory for column for all channels
            if aoesa_to_vov:  # convert to VectorOfVectors
                if col not in col_dict.keys():
                    col_dict[col] = [[]] * table_length
                for i, idx in enumerate(tcm_idx):
                    col_dict[col][idx] = tier_table[col].nda[i]
            else:  # Try to make AoESA, raise error otherwise
                if col not in col_dict.keys():
                    col_dict[col] = np.empty(
                        (table_length, len(tier_table[col].nda[0])),
                        dtype=tier_table[col].dtype,
                    )
                try:
                    col_dict[col][tcm_idx] = tier_table[col].nda
                except BaseException:
                    raise ValueError(
                        f"self.aoesa_to_vov is False but {col} is a jagged array"
                    )
        elif isinstance(tier_table[col], VectorOfVectors):
            # Allocate memory for column for all channels
            if col not in col_dict.keys():
                col_dict[col] = [[]] * table_length
            for i, idx in enumerate(tcm_idx):
                col_dict[col][idx] = tier_table[col][i]
        elif isinstance(tier_table[col], Array):
            # Allocate memory for column for all channels
            if col not in col_dict.keys():
                col_dict[col] = np.empty(
                    table_length,
                    dtype=tier_table[col].dtype,
                )
            col_dict[col][tcm_idx] = tier_table[col].nda
        elif isinstance(tier_table[col], Table):
            if col not in col_dict.keys():
                col_dict[col] = {}
            col_dict[col], attr_dict[col] = fill_col_dict(
                tier_table[col],
                col_dict[col],
                attr_dict[col],
                tcm_idx,
                table_length,
                aoesa_to_vov,
            )
        else:
            log.warning(
                f"not sure how to handle column {col} "
                f"of type {type(tier_table[col])} yet"
            )
    return col_dict, attr_dict
