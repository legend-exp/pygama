import os
from pathlib import Path

import lgdo.lh5_store as store
import numpy as np
import pytest
from lgdo import LH5Store, ls

from pygama.hit import build_hit

config_dir = Path(__file__).parent / "configs"


def test_basics(dsp_test_file, tmptestdir):
    outfile = f"{tmptestdir}/LDQTA_r117_20200110T105115Z_cal_geds_hit.lh5"

    build_hit(
        dsp_test_file,
        outfile=outfile,
        hit_config=f"{config_dir}/basic-hit-config.json",
        wo_mode="overwrite",
    )

    assert os.path.exists(outfile)
    assert ls(outfile, "/geds/") == ["geds/hit"]


def test_illegal_arguments(dsp_test_file):
    with pytest.raises(ValueError):
        build_hit(dsp_test_file)

    with pytest.raises(ValueError):
        build_hit(
            dsp_test_file,
            hit_config=f"{config_dir}/basic-hit-config.json",
            lh5_tables_config={},
        )

    with pytest.raises(ValueError):
        build_hit(
            dsp_test_file,
            lh5_tables=[],
            lh5_tables_config={},
        )


def test_lh5_table_configs(dsp_test_file, tmptestdir):
    outfile = f"{tmptestdir}/LDQTA_r117_20200110T105115Z_cal_geds_hit.lh5"

    lh5_tables_config = {"/geds/dsp": f"{config_dir}/basic-hit-config.json"}

    build_hit(
        dsp_test_file,
        outfile=outfile,
        lh5_tables_config=lh5_tables_config,
        wo_mode="overwrite",
    )

    assert os.path.exists(outfile)
    assert ls(outfile, "/geds/") == ["geds/hit"]

    lh5_tables_config = {
        "/geds/dsp": {
            "outputs": ["calE", "AoE"],
            "operations": {
                "calE": {
                    "expression": "sqrt(a + b * trapEmax**2)",
                    "parameters": {"a": 1.23, "b": 42.69},
                },
                "AoE": {"expression": "A_max/calE"},
            },
        }
    }

    build_hit(
        dsp_test_file,
        outfile=outfile,
        lh5_tables_config=lh5_tables_config,
        wo_mode="overwrite",
    )

    assert os.path.exists(outfile)
    assert ls(outfile, "/geds/") == ["geds/hit"]


def test_outputs_specification(dsp_test_file, tmptestdir):
    outfile = f"{tmptestdir}/LDQTA_r117_20200110T105115Z_cal_geds_hit.lh5"

    build_hit(
        dsp_test_file,
        outfile=outfile,
        hit_config=f"{config_dir}/basic-hit-config.json",
        wo_mode="overwrite",
    )

    store = LH5Store()
    obj, _ = store.read_object("/geds/hit", outfile)
    assert list(obj.keys()) == ["calE", "AoE", "A_max"]


def test_aggregation_outputs(dsp_test_file, tmptestdir):
    outfile = f"{tmptestdir}/LDQTA_r117_20200110T105115Z_cal_geds_hit.lh5"

    build_hit(
        dsp_test_file,
        outfile=outfile,
        hit_config=f"{config_dir}/aggregations-hit-config.json",
        wo_mode="overwrite",
    )

    sto = LH5Store()
    obj, _ = sto.read_object("/geds/hit", outfile)
    assert list(obj.keys()) == [
        "is_valid_rt",
        "is_valid_t0",
        "is_valid_tmax",
        "aggr1",
        "aggr2",
    ]

    df = store.load_dfs(
        outfile,
        ["is_valid_rt", "is_valid_t0", "is_valid_tmax", "aggr1", "aggr2"],
        "geds/hit/",
    )

    # aggr1 consists of 3 bits --> max number can be 7, aggr2 consists of 2 bits so max number can be 3
    assert not (df["aggr1"] > 7).any()
    assert not (df["aggr2"] > 3).any()

    def get_bit(x, n):
        """bit numbering from right to left, starting with bit 0"""
        return x & (1 << n) != 0

    df["bit0_check"] = df.apply(lambda row: get_bit(row["aggr1"], 0), axis=1)
    are_identical = df["bit0_check"].equals(df.is_valid_rt)
    assert are_identical

    df["bit1_check"] = df.apply(lambda row: get_bit(row["aggr1"], 1), axis=1)
    are_identical = df["bit1_check"].equals(df.is_valid_t0)
    assert are_identical

    df["bit2_check"] = df.apply(lambda row: get_bit(row["aggr1"], 2), axis=1)
    are_identical = df["bit2_check"].equals(df.is_valid_tmax)
    assert are_identical


def test_build_hit_spms_basic(dsp_test_file_spm, tmptestdir):
    out_file = f"{tmptestdir}/L200-comm-20211130-phy-spms_hit.lh5"
    build_hit(
        dsp_test_file_spm,
        outfile=out_file,
        hit_config=f"{config_dir}/spms-hit-config.json",
        wo_mode="overwrite_file",
    )
    assert ls(out_file) == ["ch0", "ch1", "ch2"]
    assert ls(out_file, "ch0/") == ["ch0/hit"]
    assert ls(out_file, "ch0/hit/") == [
        "ch0/hit/energy_in_pe",
        "ch0/hit/quality_cut",
        "ch0/hit/trigger_pos",
    ]


def test_build_hit_spms_multiconfig(dsp_test_file_spm, tmptestdir):
    out_file = f"{tmptestdir}/L200-comm-20211130-phy-spms_hit.lh5"

    build_hit(
        dsp_test_file_spm,
        outfile=out_file,
        lh5_tables_config=f"{config_dir}/spms-hit-multi-config.json",
        wo_mode="overwrite",
    )
    assert ls(out_file) == ["ch0", "ch1", "ch2"]
    assert ls(out_file, "ch0/") == ["ch0/hit"]
    assert ls(out_file, "ch0/hit/") == [
        "ch0/hit/energy_in_pe",
        "ch0/hit/quality_cut",
        "ch0/hit/trigger_pos",
    ]


def test_build_hit_spms_calc(dsp_test_file_spm, tmptestdir):
    out_file = f"{tmptestdir}/L200-comm-20211130-phy-spms_hit.lh5"

    build_hit(
        dsp_test_file_spm,
        outfile=out_file,
        wo_mode="overwrite_file",
        lh5_tables_config=f"{config_dir}/spms-hit-a-config.json",
    )
    assert ls(out_file) == ["ch0", "ch1", "ch2"]
    assert ls(out_file, "ch0/") == ["ch0/hit"]
    assert ls(out_file, "ch0/hit/") == ["ch0/hit/energy_in_pe"]

    df0 = store.load_nda(out_file, ["energy_in_pe"], "ch0/hit/")
    df1 = store.load_nda(out_file, ["energy_in_pe"], "ch1/hit/")
    df2 = store.load_nda(out_file, ["energy_in_pe"], "ch2/hit/")

    assert len(df0["energy_in_pe"]) == 5
    assert len(df1["energy_in_pe"]) == 5
    assert len(df2["energy_in_pe"]) == 5

    assert len(df0["energy_in_pe"][0]) == 20
    assert len(df1["energy_in_pe"][0]) == 20
    assert len(df2["energy_in_pe"][0]) == 20

    assert np.nanmean(df0["energy_in_pe"]) == 0
    assert np.nanmean(df1["energy_in_pe"]) == 1
    assert np.nanmean(df2["energy_in_pe"]) == 2
