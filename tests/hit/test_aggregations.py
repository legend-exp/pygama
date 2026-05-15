from __future__ import annotations

import awkward as ak
import lgdo
import numpy as np
import pytest

from pygama.hit import unpack_bitmask


def test_from_lgdo_array_uses_attr():
    arr = lgdo.Array(np.array([0b000, 0b001, 0b010, 0b101, 0b111], dtype=np.uint8))
    arr.attrs["bit_names"] = "rt,t0,tmax"

    rec = unpack_bitmask(arr)

    assert rec.fields == ["rt", "t0", "tmax"]
    assert ak.to_list(rec["rt"]) == [False, True, False, True, True]
    assert ak.to_list(rec["t0"]) == [False, False, True, False, True]
    assert ak.to_list(rec["tmax"]) == [False, False, False, True, True]


def test_explicit_bit_names_override_attr():
    arr = lgdo.Array(np.array([0b01, 0b10, 0b11], dtype=np.uint8))
    arr.attrs["bit_names"] = "a,b"

    rec = unpack_bitmask(arr, bit_names=["foo", "bar"])

    assert rec.fields == ["foo", "bar"]


def test_from_numpy_with_names():
    rec = unpack_bitmask(np.array([0b01, 0b10, 0b11]), bit_names=["low", "high"])
    assert rec.fields == ["low", "high"]
    assert ak.to_list(rec["low"]) == [True, False, True]
    assert ak.to_list(rec["high"]) == [False, True, True]


def test_from_awkward_with_names():
    rec = unpack_bitmask(ak.Array([0b01, 0b10, 0b11]), bit_names=["low", "high"])
    assert ak.to_list(rec["low"]) == [True, False, True]
    assert ak.to_list(rec["high"]) == [False, True, True]


def test_from_vector_of_vectors_uses_attr():
    vov = lgdo.VectorOfVectors([[0b01, 0b10], [0b11]])
    vov.attrs["bit_names"] = "low,high"

    rec = unpack_bitmask(vov)

    assert rec.fields == ["low", "high"]
    assert ak.to_list(rec["low"]) == [[True, False], [True]]
    assert ak.to_list(rec["high"]) == [[False, True], [True]]


def test_field_equality_filtering():
    arr = lgdo.Array(np.array([0b00, 0b01, 0b10, 0b11], dtype=np.uint8))
    arr.attrs["bit_names"] = "low,high"

    rec = unpack_bitmask(arr)

    assert ak.to_list(rec.low == True) == [False, True, False, True]  # noqa: E712
    assert ak.to_list(rec.low == False) == [True, False, True, False]  # noqa: E712
    assert ak.to_list(rec.high == True) == [False, False, True, True]  # noqa: E712

    both_set = rec[(rec.low == True) & (rec.high == True)]  # noqa: E712
    assert len(both_set) == 1


def test_lgdo_array_without_attr_raises():
    arr = lgdo.Array(np.array([1, 2, 3]))
    with pytest.raises(ValueError, match="bit_names"):
        unpack_bitmask(arr)


def test_numpy_without_names_raises():
    with pytest.raises(ValueError, match="bit_names"):
        unpack_bitmask(np.array([1, 2, 3]))
