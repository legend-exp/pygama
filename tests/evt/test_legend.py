from __future__ import annotations

import lh5
import numpy as np
import pytest
from lgdo import types

from pygama.evt import utils
from pygama.evt.modules.legend import convert_rawid

DATAINFO = utils.make_files_config({"hit": (None, None, "ch{}/hit")})
CHMAP = {"ch1104000/hit": "V14654A", "ch1104001/hit": "V03422A"}


def _convert(rawid_obj):
    return convert_rawid(DATAINFO, None, None, CHMAP, rawid_obj=rawid_obj)


def test_convert_rawid_vov():
    rawids = types.VectorOfVectors(
        flattened_data=np.array([1104000, 0, 1104001], dtype="uint32"),
        cumulative_length=np.array([2, 2, 3]),
    )
    out = _convert(rawids)

    assert out.attrs["datatype"] == "array<1>{array<1>{string}}"
    assert list(out.flattened_data.nda) == ["V14654A", "0000000", "V03422A"]
    assert list(out.cumulative_length.nda) == [2, 2, 3]


@pytest.mark.parametrize(
    ("rawids", "datatype"),
    [
        (
            types.VectorOfVectors(
                flattened_data=np.array([], dtype="uint32"),
                cumulative_length=np.array([0, 0]),
            ),
            "array<1>{array<1>{string}}",
        ),
        (types.Array(np.array([], dtype="uint32")), "array<1>{string}"),
    ],
)
def test_convert_rawid_empty(rawids, datatype, tmp_path):
    out = _convert(rawids)
    assert out.attrs["datatype"] == datatype

    # build_evt casts to the config dtype; the file must still be readable
    if isinstance(out, types.VectorOfVectors):
        out.flattened_data.nda = out.flattened_data.nda.astype("S7")
    else:
        out.nda = out.nda.astype("S7")

    fname = str(tmp_path / "evt.lh5")
    lh5.write(out, "detector_name", fname)
    assert lh5.read("detector_name", fname).attrs["datatype"] == datatype
