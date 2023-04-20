import re

import numpy as np

from pygama.lgdo import (
    Array,
    ArrayOfEqualSizedArrays,
    Table,
    VectorOfVectors,
    WaveformTable,
)


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
