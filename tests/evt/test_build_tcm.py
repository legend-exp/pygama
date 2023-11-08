import os

import lgdo
import numpy as np
from lgdo import LH5Store

from pygama import evt


def test_generate_tcm_cols(lgnd_test_data):
    f_raw = lgnd_test_data.get_path(
        "lh5/prod-ref-l200/generated/tier/raw/cal/p03/r001/l200-p03-r001-cal-20230318T012144Z-tier_raw.lh5"
    )
    tables = lgdo.ls(f_raw)
    store = LH5Store()
    coin_data = []
    for tbl in tables:
        ts, _ = store.read_object(f"{tbl}/raw/timestamp", f_raw)
        coin_data.append(ts)

    tcm_cols = evt.generate_tcm_cols(
        coin_data, 0, "last", [int(tb[2:]) for tb in tables]
    )
    assert isinstance(tcm_cols, dict)
    for v in tcm_cols.values():
        assert np.issubdtype(v.dtype, np.integer)

    # fmt: off
    assert np.array_equal(
        tcm_cols["cumulative_length"],
        [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        ],
    )
    assert np.array_equal(
        tcm_cols["array_id"],
        [
            1084804, 1084803, 1121600, 1084804, 1121600, 1084804, 1121600,
            1084804, 1084804, 1084804, 1084803, 1084804, 1084804, 1121600,
            1121600, 1084804, 1121600, 1084804, 1121600, 1084803, 1084803,
            1121600, 1121600, 1121600, 1084803, 1084803, 1084803, 1084803,
            1084803, 1084803,
        ],
    )
    assert np.array_equal(
        tcm_cols["array_idx"],
        [
            0, 0, 0, 1, 1, 2, 2, 3, 4, 5, 1, 6, 7, 3, 4, 8, 5, 9, 6, 2, 3, 7,
            8, 9, 4, 5, 6, 7, 8, 9,
        ],
    )
    # fmt: on


def test_build_tcm(lgnd_test_data, tmptestdir):
    f_raw = lgnd_test_data.get_path(
        "lh5/prod-ref-l200/generated/tier/raw/cal/p03/r001/l200-p03-r001-cal-20230318T012144Z-tier_raw.lh5"
    )
    out_file = f"{tmptestdir}/pygama-test-tcm.lh5"
    evt.build_tcm(
        [(f_raw, ["ch1084803/raw", "ch1084804/raw", "ch1121600/raw"])],
        "timestamp",
        out_file=out_file,
        out_name="hardware_tcm",
        wo_mode="of",
    )
    assert os.path.exists(out_file)
    store = LH5Store()
    obj, n_rows = store.read_object("hardware_tcm", out_file)
    assert isinstance(obj, lgdo.Struct)
    assert list(obj.keys()) == ["cumulative_length", "array_id", "array_idx"]
