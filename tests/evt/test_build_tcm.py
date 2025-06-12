import os

import lgdo
import numpy as np
import pytest
from lgdo import lh5
from lgdo.types import Table, VectorOfVectors

from pygama import evt


def test_generate_tcm_cols(lgnd_test_data):
    f_raw = lgnd_test_data.get_path(
        "lh5/prod-ref-l200/generated/tier/raw/cal/p03/r001/l200-p03-r001-cal-20230318T012144Z-tier_raw.lh5"
    )

    tcm_cols = evt.build_tcm(
        [(f_raw, f"{chan}/raw") for chan in lh5.ls(f_raw)],
        "timestamp",
        buffer_len=100,
    )

    assert isinstance(tcm_cols, Table)
    assert isinstance(tcm_cols.table_key, VectorOfVectors)
    assert isinstance(tcm_cols.row_in_table, VectorOfVectors)
    for v in tcm_cols.values():
        assert np.issubdtype(v.flattened_data.nda.dtype, np.integer)

    # test attrs
    assert set(tcm_cols.attrs.keys()) == {"datatype", "hash_func", "tables"}
    assert tcm_cols.attrs["hash_func"] == r"\d+"
    assert set(eval(tcm_cols.attrs["tables"])) == {
        f"{chan}/raw" for chan in lh5.ls(f_raw)
    }

    # fmt: off
    assert np.array_equal(
        tcm_cols.table_key.cumulative_length.nda,
        [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        ],
    )
    assert np.array_equal(
        tcm_cols.table_key.flattened_data.nda,
        [
            1084804, 1084803, 1121600, 1084804, 1121600, 1084804, 1121600,
            1084804, 1084804, 1084804, 1084803, 1084804, 1084804, 1121600,
            1121600, 1084804, 1121600, 1084804, 1121600, 1084803, 1084803,
            1121600, 1121600, 1121600, 1084803, 1084803, 1084803, 1084803,
            1084803, 1084803,
        ],
    )
    assert np.array_equal(
        tcm_cols.row_in_table.flattened_data.nda,
        [
            0, 0, 0, 1, 1, 2, 2, 3, 4, 5, 1, 6, 7, 3, 4, 8, 5, 9, 6, 2, 3, 7,
            8, 9, 4, 5, 6, 7, 8, 9,
        ],
    )

    # fmt: on
    # test with small buffer len
    tcm_cols = evt.build_tcm(
        [(f_raw, f"{chan}/raw") for chan in lh5.ls(f_raw)],
        "timestamp",
        buffer_len=1,
    )

    assert isinstance(tcm_cols, Table)
    assert isinstance(tcm_cols.table_key, VectorOfVectors)
    assert isinstance(tcm_cols.row_in_table, VectorOfVectors)
    for v in tcm_cols.values():
        assert np.issubdtype(v.flattened_data.nda.dtype, np.integer)
    # fmt: off
    assert np.array_equal(
        tcm_cols.table_key.cumulative_length.nda,
        [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        ],
    )
    assert np.array_equal(
        tcm_cols.table_key.flattened_data.nda,
        [
            1084804, 1084803, 1121600, 1084804, 1121600, 1084804, 1121600,
            1084804, 1084804, 1084804, 1084803, 1084804, 1084804, 1121600,
            1121600, 1084804, 1121600, 1084804, 1121600, 1084803, 1084803,
            1121600, 1121600, 1121600, 1084803, 1084803, 1084803, 1084803,
            1084803, 1084803,
        ],
    )
    assert np.array_equal(
        tcm_cols.row_in_table.flattened_data.nda,
        [
            0, 0, 0, 1, 1, 2, 2, 3, 4, 5, 1, 6, 7, 3, 4, 8, 5, 9, 6, 2, 3, 7,
            8, 9, 4, 5, 6, 7, 8, 9,
        ],
    )
    # fmt: on
    # test with None hash_func
    tcm_cols = evt.build_tcm(
        [(f_raw, f"{chan}/raw") for chan in lh5.ls(f_raw)],
        "timestamp",
        hash_func=None,
        buffer_len=1,
    )
    # fmt: off
    assert np.array_equal(
        tcm_cols.table_key.flattened_data.nda,
        [
            1, 0, 2, 1, 2, 1, 2, 1, 1, 1, 0,
            1, 1, 2, 2, 1, 2, 1, 2, 0, 0, 2,
            2, 2, 0, 0, 0, 0, 0, 0,
        ],
    )
    # fmt: on
    # test invalid hash func
    with pytest.raises(NotImplementedError):
        evt.build_tcm(
            [(f_raw, f"{chan}/raw") for chan in lh5.ls(f_raw)],
            "timestamp",
            hash_func=[],
        )

    # test invalid window_refs
    with pytest.raises(NotImplementedError):
        evt.build_tcm(
            [(f_raw, f"{chan}/raw") for chan in lh5.ls(f_raw)],
            "timestamp",
            window_refs="test",
        )

    # test adding extra fields
    tcm_cols = evt.build_tcm(
        [(f_raw, f"{chan}/raw") for chan in lh5.ls(f_raw)],
        "timestamp",
        out_fields="timestamp",
    )
    assert "timestamp" in tcm_cols.keys()

    # test channel appearing multiple times in single entry
    tcm_cols = evt.build_tcm(
        [(f_raw, f"{chan}/raw") for chan in lh5.ls(f_raw)],
        "timestamp",
        buffer_len=100,
        coin_windows=1,
    )
    assert np.array_equal(
        tcm_cols.table_key.cumulative_length.nda,
        [30],
    )
    # fmt: off
    assert np.array_equal(
        tcm_cols.table_key.flattened_data.nda,
        [
            1084804, 1084803, 1121600, 1084804, 1121600, 1084804,
            1121600, 1084804, 1084804, 1084804, 1084803, 1084804,
            1084804, 1121600, 1121600, 1084804, 1121600, 1084804,
            1121600, 1084803, 1084803, 1121600, 1121600, 1121600,
            1084803, 1084803, 1084803, 1084803, 1084803, 1084803,
        ],
    )
    # fmt: on


def test_build_tcm_multiple_cols(lgnd_test_data):
    f_raw = lgnd_test_data.get_path(
        "lh5/prod-ref-l200/generated/tier/raw/cal/p03/r001/l200-p03-r001-cal-20230318T012144Z-tier_raw.lh5"
    )

    with pytest.raises(ValueError):
        evt.build_tcm(
            [(f_raw, ["ch1084803/raw", "ch1084804/raw", "ch1121600/raw"])],
            coin_cols="timestamp",
            window_refs=["last", "last"],
        )
    with pytest.raises(ValueError):
        evt.build_tcm(
            [(f_raw, ["ch1084803/raw", "ch1084804/raw", "ch1121600/raw"])],
            coin_cols=["timestamp"],
            coin_windows=[1, 2],
        )
    tcm = evt.build_tcm(
        [(f_raw, ["ch1084803/raw", "ch1084804/raw", "ch1121600/raw"])],
        coin_cols=["timestamp", "table_key"],
    )
    assert isinstance(tcm, Table)
    assert isinstance(tcm.table_key, VectorOfVectors)
    assert isinstance(tcm.row_in_table, VectorOfVectors)
    # fmt: off
    assert np.array_equal(
        tcm.table_key.flattened_data.nda,
        [
            1084804, 1084803, 1121600, 1084804, 1121600, 1084804, 1121600,
            1084804, 1084804, 1084804, 1084803, 1084804, 1084804, 1121600,
            1121600, 1084804, 1121600, 1084804, 1121600, 1084803, 1084803,
            1121600, 1121600, 1121600, 1084803, 1084803, 1084803, 1084803,
            1084803, 1084803,
        ],
    )
    # fmt: on

    assert np.array_equal(
        tcm.table_key.cumulative_length.nda,
        np.arange(1, 31),
    )


def test_build_tcm_write(lgnd_test_data, tmp_dir):
    f_raw = lgnd_test_data.get_path(
        "lh5/prod-ref-l200/generated/tier/raw/cal/p03/r001/l200-p03-r001-cal-20230318T012144Z-tier_raw.lh5"
    )
    out_file = f"{tmp_dir}/pygama-test-tcm.lh5"
    evt.build_tcm(
        [(f_raw, ["ch1084803/raw", "ch1084804/raw", "ch1121600/raw"])],
        "timestamp",
        out_file=out_file,
        out_name="hardware_tcm",
        wo_mode="of",
    )

    assert os.path.exists(out_file)
    tcm_cols = lh5.read("hardware_tcm", out_file)
    assert isinstance(tcm_cols, lgdo.Struct)
    assert sorted(tcm_cols.keys()) == ["row_in_table", "table_key"]
    # fmt: off
    assert np.array_equal(
        tcm_cols.table_key.cumulative_length.nda,
        [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        ],
    )
    assert np.array_equal(
        tcm_cols.table_key.flattened_data.nda,
        [
            1084804, 1084803, 1121600, 1084804, 1121600, 1084804, 1121600,
            1084804, 1084804, 1084804, 1084803, 1084804, 1084804, 1121600,
            1121600, 1084804, 1121600, 1084804, 1121600, 1084803, 1084803,
            1121600, 1121600, 1121600, 1084803, 1084803, 1084803, 1084803,
            1084803, 1084803,
        ],
    )
    assert np.array_equal(
        tcm_cols.row_in_table.flattened_data.nda,
        [
            0, 0, 0, 1, 1, 2, 2, 3, 4, 5, 1, 6, 7, 3, 4, 8, 5, 9, 6, 2, 3, 7,
            8, 9, 4, 5, 6, 7, 8, 9,
        ],
    )
    # fmt: on
    # test also with small buffers
    evt.build_tcm(
        [(f_raw, ["ch1084803/raw", "ch1084804/raw", "ch1121600/raw"])],
        "timestamp",
        out_file=out_file,
        out_name="hardware_tcm",
        wo_mode="of",
        buffer_len=1,
    )
    assert os.path.exists(out_file)
    tcm_cols = lh5.read("hardware_tcm", out_file)
    assert isinstance(tcm_cols, lgdo.Struct)
    assert sorted(tcm_cols.keys()) == ["row_in_table", "table_key"]
    # fmt: off
    assert np.array_equal(
        tcm_cols.table_key.cumulative_length.nda,
        [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        ],
    )
    assert np.array_equal(
        tcm_cols.table_key.flattened_data.nda,
        [
            1084804, 1084803, 1121600, 1084804, 1121600, 1084804, 1121600,
            1084804, 1084804, 1084804, 1084803, 1084804, 1084804, 1121600,
            1121600, 1084804, 1121600, 1084804, 1121600, 1084803, 1084803,
            1121600, 1121600, 1121600, 1084803, 1084803, 1084803, 1084803,
            1084803, 1084803,
        ],
    )
    assert np.array_equal(
        tcm_cols.row_in_table.flattened_data.nda,
        [
            0, 0, 0, 1, 1, 2, 2, 3, 4, 5, 1, 6, 7, 3, 4, 8, 5, 9, 6, 2, 3, 7,
            8, 9, 4, 5, 6, 7, 8, 9,
        ],
    )
    # fmt: on
    # test also with small buffers
    evt.build_tcm(
        [(f_raw, ["ch1084803/raw", "ch1084804/raw", "ch1121600/raw"])],
        "timestamp",
        out_file=out_file,
        out_name="hardware_tcm",
        wo_mode="of",
        buffer_len=1,
    )
    assert os.path.exists(out_file)
    tcm_cols = lh5.read("hardware_tcm", out_file)
    assert isinstance(tcm_cols, lgdo.Struct)
    assert sorted(tcm_cols.keys()) == ["row_in_table", "table_key"]
    # fmt: off
    assert np.array_equal(
        tcm_cols.table_key.cumulative_length.nda,
        [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        ],
    )
    assert np.array_equal(
        tcm_cols.table_key.flattened_data.nda,
        [
            1084804, 1084803, 1121600, 1084804, 1121600, 1084804, 1121600,
            1084804, 1084804, 1084804, 1084803, 1084804, 1084804, 1121600,
            1121600, 1084804, 1121600, 1084804, 1121600, 1084803, 1084803,
            1121600, 1121600, 1121600, 1084803, 1084803, 1084803, 1084803,
            1084803, 1084803,
        ],
    )
    assert np.array_equal(
        tcm_cols.row_in_table.flattened_data.nda,
        [
            0, 0, 0, 1, 1, 2, 2, 3, 4, 5, 1, 6, 7, 3, 4, 8, 5, 9, 6, 2, 3, 7,
            8, 9, 4, 5, 6, 7, 8, 9,
        ],
    )
    # fmt: on
