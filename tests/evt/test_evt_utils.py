from __future__ import annotations

import types

import awkward as ak
import lh5
import numpy as np
import pytest
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


def test_make_numpy_full_promotes_to_hold_both():
    """The accumulator must hold the fill value *and* the data dtype.

    A config ``initial: 0`` against float32 data cannot be cast to float32;
    returning an integer accumulator makes the later ``out += res`` raise.
    """
    # the regression: integer initial, float32 data
    out = utils.make_numpy_full(4, 0, np.float32)
    assert out.dtype == np.float32
    res = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
    out[np.arange(4)] += res  # must not raise UFuncOutputCastingError
    assert out.tolist() == [1.5, 2.5, 3.5, 4.5]

    # float64 data keeps working exactly as before
    assert utils.make_numpy_full(3, 0, np.float64).dtype == np.float64
    # bool initial with bool data stays bool
    assert utils.make_numpy_full(3, False, bool).dtype == np.bool_
    # a nan initial forces a float accumulator even against integer data
    assert np.issubdtype(utils.make_numpy_full(3, np.nan, np.int32).dtype, np.floating)
    # the value is still the requested one
    assert np.all(utils.make_numpy_full(3, 7, np.float32) == 7)


def test_make_numpy_full_accepts_python_types():
    """``evaluate_to_first_or_last`` passes ``type(default_value)`` as the dtype."""
    assert utils.make_numpy_full(2, 0, int).dtype == np.dtype(int)
    assert utils.make_numpy_full(2, 0.0, float).dtype == np.dtype(float)


def test_channel_indices_rejects_an_unresolved_table_id():
    """A channel name that does not match the table format must fail clearly.

    ``get_tcm_id_by_pattern`` returns ``None`` for such names, and both the
    cached and uncached paths would otherwise die inside awkward with an
    opaque "None conversion/promotion is disabled" TypeError.
    """
    tcm = types.SimpleNamespace(
        table_key=ak.Array([[1, 2], [3]]),
        row_in_table=ak.Array([[0, 1], [2]]),
    )

    # uncached path
    with pytest.raises(ValueError, match="does not match the table format"):
        utils.channel_indices(tcm, None)

    # and with a cache attached, so both routes agree
    tcm.cache = utils.EvtCache()
    with pytest.raises(ValueError, match="does not match the table format"):
        utils.channel_indices(tcm, None)

    # a resolvable id still works
    assert utils.get_tcm_id_by_pattern("ch{}", "ch3") == 3
    assert utils.get_tcm_id_by_pattern("ch{}", "not-a-channel") is None
