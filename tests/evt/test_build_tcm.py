from __future__ import annotations

from pathlib import Path

import awkward as ak
import lgdo
import lh5
import numpy as np
import pytest
from lgdo.types import Table, VectorOfVectors

from pygama import evt


def test_generate_tcm_cols(lgnd_test_data):
    f_raw = lgnd_test_data.get_path(
        "lh5/prod-ref-l200/generated/tier/raw/cal/p03/r001/l200-p03-r001-cal-20230318T012144Z-tier_raw.lh5"
    )

    chan_list = lh5.ls(f_raw)
    tcm_cols = evt.build_tcm(
        [(f_raw, [f"{chan}/raw" for chan in chan_list])],
        "timestamp",
        buffer_len=100,
        channel_views=None,
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
    exp_keys = VectorOfVectors(
        cumulative_length = np.array(
            [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20,
              21, 22, 23, 24, 25, 26, 27, 28, 29, 30, ],
        ),
        flattened_data = np.array(
            [ 1084804, 1084803, 1121600, 1084804, 1121600, 1084804, 1121600,
              1084804, 1084804, 1084804, 1084803, 1084804, 1084804, 1121600,
              1121600, 1084804, 1121600, 1084804, 1121600, 1084803, 1084803,
              1121600, 1121600, 1121600, 1084803, 1084803, 1084803, 1084803,
              1084803, 1084803, ]
        )
    )
    exp_rows = VectorOfVectors(
        cumulative_length = np.array(
            [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20,
              21, 22, 23, 24, 25, 26, 27, 28, 29, 30, ],
        ),
        flattened_data = np.array(
            [ 0, 0, 0, 1, 1, 2, 2, 3, 4, 5, 1, 6, 7, 3, 4, 8, 5, 9, 6, 2, 3, 7,
              8, 9, 4, 5, 6, 7, 8, 9, ]
        )
    )

    # fmt: on
    assert tcm_cols.table_key == exp_keys
    assert tcm_cols.row_in_table == exp_rows

    # Test with sparse views enabled
    (tcm_cols2, chan_tcms) = evt.build_tcm(
        [(f_raw, "ch*/raw")],
        "timestamp",
        buffer_len=100,
        channel_views="sparse",
    )

    assert tcm_cols2.table_key == exp_keys
    assert tcm_cols2.row_in_table == exp_rows

    assert set(chan_list) == set(chan_tcms.keys())
    assert all(np.issubdtype(v.dtype, np.integer) for v in chan_tcms.values())
    for chan_name, entries in chan_tcms.items():
        chan_id = int(chan_name[2:])
        assert np.all(
            entries
            == np.flatnonzero(
                np.any(tcm_cols2.table_key.view_as("ak") == chan_id, axis=-1)
            )
        )

    # test with small buffer len
    tcm_cols = evt.build_tcm(
        [(f_raw, [f"{chan}/raw" for chan in lh5.ls(f_raw)])],
        "timestamp",
        buffer_len=1,
        channel_views=None,
    )

    assert tcm_cols.table_key == exp_keys
    assert tcm_cols.row_in_table == exp_rows

    # test with None hash_func
    tcm_cols = evt.build_tcm(
        [(f_raw, [f"{chan}/raw" for chan in lh5.ls(f_raw)])],
        "timestamp",
        hash_func=None,
        buffer_len=1,
        channel_views=None,
    )
    # fmt: off
    exp_idxs = VectorOfVectors(
        cumulative_length = np.array(
            [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20,
              21, 22, 23, 24, 25, 26, 27, 28, 29, 30, ],
        ),
        flattened_data = np.array(
            [ 1, 0, 2, 1, 2, 1, 2, 1, 1, 1, 0,
              1, 1, 2, 2, 1, 2, 1, 2, 0, 0, 2,
              2, 2, 0, 0, 0, 0, 0, 0, ],
        )
    )
    # fmt: on
    assert tcm_cols.table_key == exp_idxs
    assert tcm_cols.row_in_table == exp_rows

    # test with None hash_func, and with channel_views
    (tcm_cols2, chan_tcms) = evt.build_tcm(
        [(f_raw, [f"{chan}/raw" for chan in lh5.ls(f_raw)])],
        "timestamp",
        hash_func=None,
        buffer_len=1,
        channel_views="sparse",
    )

    assert tcm_cols2.table_key == exp_idxs
    assert tcm_cols2.row_in_table == exp_rows

    assert {f"ch{i}" for i in range(3)} == set(chan_tcms.keys())
    assert all(np.issubdtype(v.dtype, np.integer) for v in chan_tcms.values())
    for chan_name, entries in chan_tcms.items():
        chan_id = int(chan_name[2:])
        assert np.all(
            entries
            == np.flatnonzero(
                np.any(tcm_cols2.table_key.view_as("ak") == chan_id, axis=-1)
            )
        )

    # test invalid hash func
    with pytest.raises(NotImplementedError):
        evt.build_tcm(
            [(f_raw, [f"{chan}/raw" for chan in lh5.ls(f_raw)])],
            "timestamp",
            hash_func=[],
        )

    # test invalid window_refs
    with pytest.raises(NotImplementedError):
        evt.build_tcm(
            [(f_raw, [f"{chan}/raw" for chan in lh5.ls(f_raw)])],
            "timestamp",
            window_refs="test",
        )

    # test adding extra fields
    tcm_cols = evt.build_tcm(
        [(f_raw, [f"{chan}/raw" for chan in lh5.ls(f_raw)])],
        "timestamp",
        out_fields="timestamp",
        channel_views=None,
    )
    assert "timestamp" in tcm_cols

    # test channel appearing multiple times in single entry
    tcm_cols = evt.build_tcm(
        [(f_raw, [f"{chan}/raw" for chan in lh5.ls(f_raw)])],
        "timestamp",
        buffer_len=100,
        coin_windows=1,
        channel_views=None,
    )
    # fmt: off
    exp_keys = VectorOfVectors(
        flattened_data=np.array(
            [ 1084804, 1084803, 1121600, 1084804, 1121600, 1084804,
              1121600, 1084804, 1084804, 1084804, 1084803, 1084804,
              1084804, 1121600, 1121600, 1084804, 1121600, 1084804,
              1121600, 1084803, 1084803, 1121600, 1121600, 1121600,
              1084803, 1084803, 1084803, 1084803, 1084803, 1084803, ]
        ),
        cumulative_length=np.array([30])
    )
    # fmt: on
    assert tcm_cols.table_key == exp_keys

    # test channel appearing multiple times in single entry
    tcm_cols, chan_tcms = evt.build_tcm(
        [(f_raw, [f"{chan}/raw" for chan in lh5.ls(f_raw)])],
        "timestamp",
        buffer_len=100,
        coin_windows=1,
        channel_views="sparse",
    )
    assert tcm_cols.table_key == exp_keys
    assert set(chan_list) == set(chan_tcms.keys())
    for entries in chan_tcms.values():
        assert (entries == np.array([0])).all()


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

    attrs = tcm.attrs
    assert "hash_func" in attrs
    assert "tables" in attrs
    assert attrs["tables"] == "['ch1084803/raw', 'ch1084804/raw', 'ch1121600/raw']"


def test_build_tcm_write(lgnd_test_data, tmp_dir):
    f_raw = lgnd_test_data.get_path(
        "lh5/prod-ref-l200/generated/tier/raw/cal/p03/r001/l200-p03-r001-cal-20230318T012144Z-tier_raw.lh5"
    )
    out_file = f"{tmp_dir}/pygama-test-tcm.lh5"
    channels = ["ch1084803", "ch1084804", "ch1121600"]
    evt.build_tcm(
        [(f_raw, [f"{ch}/raw" for ch in channels])],
        "timestamp",
        out_file=out_file,
        out_name="hardware_tcm",
        wo_mode="of",
        channel_views=None,
    )

    assert Path(out_file).exists()
    tcm_cols = lh5.read("hardware_tcm", out_file)
    assert isinstance(tcm_cols, lgdo.Struct)
    assert sorted(tcm_cols.keys()) == ["row_in_table", "table_key"]
    # fmt: off
    exp_keys = VectorOfVectors(
        cumulative_length = np.array(
            [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20,
              21, 22, 23, 24, 25, 26, 27, 28, 29, 30, ],
        ),
        flattened_data = np.array(
            [ 1084804, 1084803, 1121600, 1084804, 1121600, 1084804, 1121600,
              1084804, 1084804, 1084804, 1084803, 1084804, 1084804, 1121600,
              1121600, 1084804, 1121600, 1084804, 1121600, 1084803, 1084803,
              1121600, 1121600, 1121600, 1084803, 1084803, 1084803, 1084803,
              1084803, 1084803, ]
        )
    )
    exp_rows = VectorOfVectors(
        cumulative_length = np.array(
            [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20,
              21, 22, 23, 24, 25, 26, 27, 28, 29, 30, ],
        ),
        flattened_data = np.array(
            [ 0, 0, 0, 1, 1, 2, 2, 3, 4, 5, 1, 6, 7, 3, 4, 8, 5, 9, 6, 2, 3, 7,
              8, 9, 4, 5, 6, 7, 8, 9, ]
        )
    )
    # fmt: on

    assert tcm_cols.table_key == exp_keys
    assert tcm_cols.row_in_table == exp_rows

    # test with views also written
    evt.build_tcm(
        [(f_raw, [f"{ch}/raw" for ch in channels])],
        "timestamp",
        out_file=out_file,
        out_name="hardware_tcm",
        wo_mode="of",
        channel_views="sparse",
    )
    assert Path(out_file).exists()
    tcm_cols = lh5.read("hardware_tcm", out_file)
    assert isinstance(tcm_cols, lgdo.Struct)
    assert sorted(tcm_cols.keys()) == ["row_in_table", "table_key"]
    assert tcm_cols.table_key == exp_keys
    assert tcm_cols.row_in_table == exp_rows

    for ch in channels:
        ch_tcm = lh5.read(f"{ch}/hardware_tcm", out_file)
        ch_id = int(ch[2:])
        mask = ak.any(tcm_cols.table_key.view_as("ak") == ch_id, axis=-1)
        assert ch_tcm == tcm_cols[mask]

    # Test both view modes with small buffers in a fresh output file.
    small_out_file = f"{tmp_dir}/pygama-test-tcm-small-buffer.lh5"
    evt.build_tcm(
        [(f_raw, [f"{ch}/raw" for ch in channels])],
        "timestamp",
        out_file=small_out_file,
        out_name="hardware_tcm",
        wo_mode="of",
        buffer_len=1,
        channel_views=None,
    )
    assert Path(small_out_file).exists()
    tcm_cols = lh5.read("hardware_tcm", small_out_file)
    assert isinstance(tcm_cols, lgdo.Struct)
    assert sorted(tcm_cols.keys()) == ["row_in_table", "table_key"]

    assert tcm_cols.table_key == exp_keys
    assert tcm_cols.row_in_table == exp_rows

    # test with sparse views and small buffers
    evt.build_tcm(
        [(f_raw, [f"{ch}/raw" for ch in channels])],
        "timestamp",
        out_file=small_out_file,
        out_name="hardware_tcm",
        wo_mode="of",
        buffer_len=1,
        channel_views="sparse",
    )
    assert Path(small_out_file).exists()
    tcm_cols = lh5.read("hardware_tcm", small_out_file)
    assert isinstance(tcm_cols, lgdo.Struct)
    assert sorted(tcm_cols.keys()) == ["row_in_table", "table_key"]
    assert tcm_cols.table_key == exp_keys
    assert tcm_cols.row_in_table == exp_rows

    for ch in channels:
        ch_tcm = lh5.read(f"{ch}/hardware_tcm", small_out_file)
        ch_id = int(ch[2:])
        mask = ak.any(tcm_cols.table_key.view_as("ak") == ch_id, axis=-1)
        assert ch_tcm == tcm_cols[mask]

    # test append to input file
    clone = f"{tmp_dir}/test-append-tcm-input.lh5"
    tables = ["ch1084803/raw", "ch1084804/raw", "ch1121600/raw"]

    with lh5.LH5Store(keep_open=True, default_mode="of") as st:
        for table in tables:
            st.write(lh5.read(table, f_raw), table, clone)

    evt.build_tcm(
        [(clone, tables)],
        "timestamp",
        out_file=clone,
        out_name="/tcm",
        wo_mode="a",
        buffer_len=1,
    )
    assert Path(clone).exists()
    tcm_cols = lh5.read("tcm", clone)
    assert isinstance(tcm_cols, lgdo.Struct)
    assert sorted(tcm_cols.keys()) == ["row_in_table", "table_key"]


def test_build_tcm_multiple_files(lgnd_test_data, tmp_dir):  # noqa: ARG001
    f_raw = lgnd_test_data.get_path(
        "lh5/prod-ref-l200/generated/tier/raw/cal/p03/r001/l200-p03-r001-cal-20230318T012144Z-tier_raw.lh5"
    )

    tcm_orig = evt.build_tcm(
        [
            (f_raw, ["ch1084803/raw", "ch1084804/raw", "ch1121600/raw"]),
        ],
        coin_cols="timestamp",
    ).view_as("ak")

    tcm = evt.build_tcm(
        [
            (f_raw, ["ch1084803/raw"]),
            (f_raw, ["ch1084804/raw", "ch1121600/raw"]),
        ],
        coin_cols="timestamp",
    ).view_as("ak")

    assert tcm_orig.fields == tcm.fields
    assert all(ak.all(tcm_orig[f] == tcm[f]) for f in tcm.fields)


def test_build_tcm_buffer_reuse_copy_regression(lgnd_test_data):
    """
    Regression test for LH5Iterator's buffer reuse: build_tcm must not retain
    views into per-chunk buffers across iterations.
    """
    f_raw = lgnd_test_data.get_path(
        "lh5/prod-ref-l200/generated/tier/raw/cal/p03/r001/l200-p03-r001-cal-20230318T012144Z-tier_raw.lh5"
    )

    # small buffer_len forces multiple iterator iterations, which used to expose
    # corruption if per-chunk buffers weren't copied before concatenation/merge.
    tcm_small = evt.build_tcm(
        [(f_raw, ["ch1084803/raw", "ch1084804/raw", "ch1121600/raw"])],
        coin_cols="timestamp",
        buffer_len=1,
    ).view_as("ak")

    tcm_large = evt.build_tcm(
        [(f_raw, ["ch1084803/raw", "ch1084804/raw", "ch1121600/raw"])],
        coin_cols="timestamp",
        buffer_len=1_000_000,
    ).view_as("ak")

    assert tcm_small.fields == tcm_large.fields
    assert all(ak.all(tcm_small[f] == tcm_large[f]) for f in tcm_small.fields)


# Test with phy data (non-sparse mode); most important for views in dense/all modes
def test_build_tcm_phy(lgnd_test_data):
    f_raw = lgnd_test_data.get_path(
        "lh5/prod-ref-l200/generated/tier/raw/phy/p03/r001/l200-p03-r001-phy-20230322T160139Z-tier_raw.lh5"
    )
    channels = ["ch1084803", "ch1084804", "ch1121600"]

    tcm_cols = evt.build_tcm(
        [(f_raw, [f"{ch}/raw" for ch in channels])],
        "timestamp",
        out_name="tcm",
        wo_mode="of",
        channel_views=None,
    )

    # fmt: off
    exp_keys = VectorOfVectors(
        cumulative_length = np.arange(1, 11)*3,
        flattened_data = np.tile([1084803,1084804,1121600], 10)
    )
    exp_rows = VectorOfVectors(
        cumulative_length = np.arange(1, 11)*3,
        flattened_data = np.repeat(np.arange(10), 3)
    )
    # fmt: on

    assert tcm_cols.table_key == exp_keys
    assert tcm_cols.row_in_table == exp_rows

    # Now test with views in "all" mode
    (tcm_cols, chan_tcms) = evt.build_tcm(
        [(f_raw, [f"{ch}/raw" for ch in channels])],
        "timestamp",
        out_name="tcm",
        wo_mode="of",
        channel_views="all",
    )

    assert tcm_cols.table_key == exp_keys
    assert tcm_cols.row_in_table == exp_rows

    assert set(channels) == set(chan_tcms.keys())
    assert all(np.issubdtype(v.dtype, np.integer) for v in chan_tcms.values())
    assert all(np.all(v == np.array([0, 10])) for v in chan_tcms.values())

    # Now test with views in "all" mode with short buffer
    (tcm_cols, chan_tcms) = evt.build_tcm(
        [(f_raw, [f"{ch}/raw" for ch in channels])],
        "timestamp",
        out_name="tcm",
        wo_mode="of",
        buffer_len=1,
        channel_views="all",
    )

    assert tcm_cols.table_key == exp_keys
    assert tcm_cols.row_in_table == exp_rows

    assert set(channels) == set(chan_tcms.keys())
    assert all(np.issubdtype(v.dtype, np.integer) for v in chan_tcms.values())
    assert all(np.all(v == np.array([0, 10])) for v in chan_tcms.values())

    # Now test with views in "dense" mode
    (tcm_cols, chan_tcms) = evt.build_tcm(
        [(f_raw, [f"{ch}/raw" for ch in channels])],
        "timestamp",
        out_name="tcm",
        wo_mode="of",
        channel_views="dense",
    )

    assert tcm_cols.table_key == exp_keys
    assert tcm_cols.row_in_table == exp_rows

    assert set(channels) == set(chan_tcms.keys())
    assert all(np.issubdtype(v.dtype, np.integer) for v in chan_tcms.values())
    assert all(np.all(v == np.array([0, 10])) for v in chan_tcms.values())

    # Now test with views in "dense" mode with short buffer
    (tcm_cols, chan_tcms) = evt.build_tcm(
        [(f_raw, [f"{ch}/raw" for ch in channels])],
        "timestamp",
        out_name="tcm",
        wo_mode="of",
        buffer_len=1,
        channel_views="dense",
    )
    assert tcm_cols.table_key == exp_keys
    assert tcm_cols.row_in_table == exp_rows

    assert set(channels) == set(chan_tcms.keys())
    assert all(np.issubdtype(v.dtype, np.integer) for v in chan_tcms.values())
    assert all(
        np.all(v == np.stack([np.arange(10), np.arange(1, 11)], axis=1))
        for v in chan_tcms.values()
    )


# Test with phy data (non-sparse mode); most important for views in dense/all modes
def test_build_tcm_write_phy(lgnd_test_data, tmp_dir):
    f_raw = lgnd_test_data.get_path(
        "lh5/prod-ref-l200/generated/tier/raw/phy/p03/r001/l200-p03-r001-phy-20230322T160139Z-tier_raw.lh5"
    )
    out_file = f"{tmp_dir}/pygama-test-tcm-phy.lh5"
    channels = ["ch1084803", "ch1084804", "ch1121600"]

    evt.build_tcm(
        [(f_raw, [f"{ch}/raw" for ch in channels])],
        "timestamp",
        out_file=out_file,
        out_name="tcm",
        wo_mode="of",
        channel_views=None,
    )

    assert Path(out_file).exists()
    tcm_cols = lh5.read("tcm", out_file)

    # fmt: off
    exp_keys = VectorOfVectors(
        cumulative_length = np.arange(1, 11)*3,
        flattened_data = np.tile([1084803,1084804,1121600], 10)
    )
    exp_rows = VectorOfVectors(
        cumulative_length = np.arange(1, 11)*3,
        flattened_data = np.repeat(np.arange(10), 3)
    )
    # fmt: on

    assert tcm_cols.table_key == exp_keys
    assert tcm_cols.row_in_table == exp_rows

    # Now test with views in "all" mode
    evt.build_tcm(
        [(f_raw, [f"{ch}/raw" for ch in channels])],
        "timestamp",
        out_file=out_file,
        out_name="tcm",
        wo_mode="of",
        channel_views="all",
    )
    assert Path(out_file).exists()
    tcm_cols = lh5.read("tcm", out_file)
    assert tcm_cols.table_key == exp_keys
    assert tcm_cols.row_in_table == exp_rows

    for ch in channels:
        ch_tcm = lh5.read(f"{ch}/tcm", out_file)
        assert ch_tcm.table_key == exp_keys
        assert ch_tcm.row_in_table == exp_rows

    # Now test with views in "all" mode with short buffer
    evt.build_tcm(
        [(f_raw, [f"{ch}/raw" for ch in channels])],
        "timestamp",
        out_file=out_file,
        out_name="tcm",
        wo_mode="of",
        buffer_len=1,
        channel_views="all",
    )
    assert Path(out_file).exists()
    tcm_cols = lh5.read("tcm", out_file)
    assert tcm_cols.table_key == exp_keys
    assert tcm_cols.row_in_table == exp_rows

    for ch in channels:
        ch_tcm = lh5.read(f"{ch}/tcm", out_file)
        assert ch_tcm.table_key == exp_keys
        assert ch_tcm.row_in_table == exp_rows

    # Now test with views in "dense" mode
    evt.build_tcm(
        [(f_raw, [f"{ch}/raw" for ch in channels])],
        "timestamp",
        out_file=out_file,
        out_name="tcm",
        wo_mode="of",
        channel_views="dense",
    )
    assert Path(out_file).exists()
    tcm_cols = lh5.read("tcm", out_file)
    assert tcm_cols.table_key == exp_keys
    assert tcm_cols.row_in_table == exp_rows

    for ch in channels:
        ch_tcm = lh5.read(f"{ch}/tcm", out_file)
        assert ch_tcm.table_key == exp_keys
        assert ch_tcm.row_in_table == exp_rows

    # Now test with views in "dense" mode with short buffer
    evt.build_tcm(
        [(f_raw, [f"{ch}/raw" for ch in channels])],
        "timestamp",
        out_file=out_file,
        out_name="tcm",
        wo_mode="of",
        buffer_len=1,
        channel_views="dense",
    )
    assert Path(out_file).exists()
    tcm_cols = lh5.read("tcm", out_file)
    assert tcm_cols.table_key == exp_keys
    assert tcm_cols.row_in_table == exp_rows

    for ch in channels:
        ch_tcm = lh5.read(f"{ch}/tcm", out_file)
        assert ch_tcm.table_key == exp_keys
        assert ch_tcm.row_in_table == exp_rows
