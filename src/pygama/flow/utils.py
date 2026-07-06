from __future__ import annotations

from dbetto import TextDB
import keyword
import logging
import re
import string
from datetime import datetime, timezone

import awkward as ak
import numpy as np
import pandas as pd
from lgdo import Array, ArrayOfEqualSizedArrays, Table, VectorOfVectors, WaveformTable

log = logging.getLogger(__name__)

# Util functions associated with dataloader

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
                col_dict[col][tcm_idx] = tier_table[col].nda
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

# Util functions associated with queries

def get_recursive(db: TextDB, path: str):
    """Helper to recursively access values from nested dict-likes"""
    fields = re.split("[,/]", path)
    ret = db
    for f in fields:
        ret = ret[f]
    return ret

def format_vars(fstring: str):
    """Helper to get list of variables referenced in format string"""
    return [v[1] for v in string.Formatter().parse(fstring) if v[1]]


def parse_query_paths(
    expr: str, fullmatch: bool = False
) -> list[tuple[str, str | None, str]] | tuple[str, str | None, str]:
    """
    Parse input string for variable names of the form::

        [alias][@ or :][par.path]

    and return a list of each matching 3-tuple of the form::

        (full_match, alias, path)

    Aliases and names in paths must be legal python names (i.e. alphanumeric, doesn't
    start with a digit). If ``@`` is used to separate the alias and path, it is left in
    the path (to denote a metadata location); if ``:`` is used, it is omitted.
    Note that function names (i.e. a valid name followed by ``(``) are excluded.
    Values inside of ``[...]``, ``{...}``, ``"..."``, and ``'...'`` are also excluded.

    If fullmatch is ``True``, expect full string to match pattern and return single tuple.
    Otherwise return a list of tuples, for each match found.
    """
    # Note: ast does not like @'s and :'s used in this way, so instead we parse with regex
    if not fullmatch:
        # remove substrings inside of brackets or quotes
        var_list = " ".join(
            re.split(r"(?:\{.*?\})|(?:\[.*?\])|(?:\".*?\")|(?:'.*?')", expr)
        )
        var_list = re.findall(r"[\w:@\.]+(?![\w:@\.(])", var_list)
    else:
        var_list = [expr]

    ret = []
    for var in var_list:
        # skip numerals
        try:
            float(var)
            if fullmatch:
                msg = f"'{var}' is not a valid variable"
                raise NameError(msg)
            continue
        except ValueError:
            pass

        # skip reserved keywords in python
        if keyword.iskeyword(var):
            if fullmatch:
                msg = f"'{var}' is an illegal name"
                raise NameError(msg)
            continue

        match = re.fullmatch(
            r"([a-zA-Z_]\w*)??:?(@?[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*)", var
        )
        if match is None:
            msg = f"'{var}' could not be parsed"
            raise NameError(msg)

        if keyword.iskeyword(match.group(1)):
            msg = f"{match.group(1)} is an illegal name"
            raise NameError(msg)
        ret.append((match.group(0), match.group(1), match.group(2)))

    return ret if not fullmatch else ret[0]
