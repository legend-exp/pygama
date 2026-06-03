from __future__ import annotations

import lh5
import numpy as np
from lgdo import Array

from pygama.evt import utils


def test_tier_data_tuple():
    files = utils.make_files_config(
        {
            "tcm": ("f1", "g1"),
            "dsp": ("f2", "g2"),
            "hit": ("f3", "g3"),
            "evt": ("f4", "g4"),
        }
    )

    assert files.raw == utils.H5DataLoc()
    assert files.tcm.file == "f1"
    assert files.tcm.group == "g1"
    assert files.dsp.file == "f2"
    assert files.dsp.group == "g2"
    assert files.hit.file == "f3"
    assert files.hit.group == "g3"
    assert files.evt.file == "f4"
    assert files.evt.group == "g4"


def test_get_lgdo_attrs(tmp_path):
    hit_file = str(tmp_path / "hit.lh5")
    arr = Array(np.array([0b01, 0b10], dtype=np.uint8))
    arr.attrs["bit_names"] = "low,high"
    lh5.write(arr, "ch1000000/hit/flags", hit_file)

    datainfo = utils.make_files_config(
        {
            "tcm": (None, "tcm"),
            "hit": (hit_file, "hit"),
            "evt": (None, "evt"),
        }
    )
    attrs = utils.get_lgdo_attrs(datainfo, ["ch1000000"], "hit", "flags")
    assert attrs.get("bit_names") == "low,high"
    assert "datatype" not in attrs

    # non-existent channel is skipped; valid one still returns attrs
    attrs = utils.get_lgdo_attrs(datainfo, ["ch9999999", "ch1000000"], "hit", "flags")
    assert attrs.get("bit_names") == "low,high"

    # unknown tier returns an empty dict rather than raising
    assert utils.get_lgdo_attrs(datainfo, ["ch1000000"], "dsp", "flags") == {}
