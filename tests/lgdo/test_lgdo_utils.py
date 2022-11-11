import os

import numpy as np
import pytest

import pygama.lgdo.lgdo_utils as lgdo_utils


def test_get_element_type():

    objs = [
        ("hi", "string"),
        (True, "bool"),
        (np.void(0), "blob"),
        (int(0), "real"),
        (np.uint8(0), "real"),
        (float(0), "real"),
        (1 + 1j, "complex"),
        (b"hi", "string"),
        (np.array(["hi"]), "string"),
    ]

    for obj, name in objs:
        get_name = lgdo_utils.get_element_type(obj)
        assert get_name == name


def test_parse_datatype():

    datatypes = [
        ("real", ("scalar", None, "real")),
        ("array<1>{bool}", ("array", (1,), "bool")),
        ("fixedsizearray<2>{real}", ("fixedsizearray", (2,), "real")),
        (
            "arrayofequalsizedarrays<3,4>{complex}",
            ("arrayofequalsizedarrays", (3, 4), "complex"),
        ),
        ("array<1>{array<1>{blob}}", ("array", (1,), "array<1>{blob}")),
        (
            "struct{field1,field2,fieldn}",
            ("struct", None, ["field1", "field2", "fieldn"]),
        ),
        ("table{col1,col2,coln}", ("table", None, ["col1", "col2", "coln"])),
    ]

    for string, dt_tuple in datatypes:
        pd_dt_tuple = lgdo_utils.parse_datatype(string)
        assert pd_dt_tuple == dt_tuple


def test_expand_path(lgnd_test_data):
    files = [
        lgnd_test_data.get_path(
            "lh5/prod-ref-l200/generated/tier/dsp/cal/p01/r014/l60-p01-r014-cal-20220716T104550Z-tier_dsp.lh5"
        ),
        lgnd_test_data.get_path(
            "lh5/prod-ref-l200/generated/tier/dsp/cal/p01/r014/l60-p01-r014-cal-20220716T105236Z-tier_dsp.lh5"
        ),
    ]
    base_dir = os.path.dirname(files[0])

    assert lgdo_utils.expand_path(f"{base_dir}/*20220716T104550Z*") == files[0]

    # Should fail if file not found
    with pytest.raises(FileNotFoundError):
        lgdo_utils.expand_path(f"{base_dir}/not_a_real_file.lh5")

    # Should fail if multiple files found
    with pytest.raises(FileNotFoundError):
        lgdo_utils.expand_path(f"{base_dir}/*.lh5")

    # Check if it finds a list of files correctly
    assert sorted(lgdo_utils.expand_path(f"{base_dir}/*.lh5", True)) == sorted(files)
