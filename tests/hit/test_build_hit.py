import json
import os
from pathlib import Path

import awkward as ak
import numpy as np
import pytest
from lgdo import lh5

from pygama.hit import build_hit
from pygama.hit.build_hit import _reorder_table_operations

config_dir = Path(__file__).parent / "configs"


def test_ops_reorder():
    assert list(_reorder_table_operations({}).keys()) == []

    ops = {
        "out1": {"expression": "out2 + out3 * outy"},
        "out2": {"expression": "log(out4)"},
        "out3": {"expression": "outx + 2"},
        "out4": {"expression": "outz + out3"},
    }
    assert list(_reorder_table_operations(ops).keys()) == [
        "out3",
        "out4",
        "out2",
        "out1",
    ]


def test_basics(dsp_test_file, tmp_dir):
    outfile = f"{tmp_dir}/test_cal_geds_hit.lh5"

    build_hit(
        dsp_test_file,
        outfile=outfile,
        hit_config=f"{config_dir}/basic-hit-config.json",
        wo_mode="overwrite",
    )

    assert os.path.exists(outfile)
    assert lh5.ls(outfile, "/ch1084803/") == ["ch1084803/hit"]

    tbl = lh5.read("ch1084803/hit", outfile)
    assert tbl.calE.attrs == {"datatype": "array<1>{real}", "units": "keV"}


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


def test_lh5_table_configs(dsp_test_file, tmp_dir):
    outfile = f"{tmp_dir}/test_cal_geds_hit.lh5"

    lh5_tables_config = {"/ch1084803/dsp": f"{config_dir}/basic-hit-config.json"}

    build_hit(
        dsp_test_file,
        outfile=outfile,
        lh5_tables_config=lh5_tables_config,
        wo_mode="overwrite",
    )

    assert os.path.exists(outfile)
    assert lh5.ls(outfile, "/ch1084803/") == ["ch1084803/hit"]

    lh5_tables_config = {
        "/ch1084803/dsp": {
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
    assert lh5.ls(outfile, "/ch1084803/") == ["ch1084803/hit"]


def test_outputs_specification(dsp_test_file, tmp_dir):
    outfile = f"{tmp_dir}/test_cal_geds_hit.lh5"

    build_hit(
        dsp_test_file,
        outfile=outfile,
        hit_config=f"{config_dir}/basic-hit-config.json",
        wo_mode="overwrite",
    )

    obj = lh5.read("/ch1084803/hit", outfile)
    assert sorted(obj.keys()) == ["A_max", "AoE", "calE"]


def test_aggregation_outputs(dsp_test_file, tmp_dir):
    outfile = f"{tmp_dir}/test_cal_geds_hit.lh5"

    build_hit(
        dsp_test_file,
        outfile=outfile,
        hit_config=f"{config_dir}/aggregations-hit-config.json",
        wo_mode="overwrite",
    )

    obj = lh5.read("/ch1084803/hit", outfile)
    assert sorted(obj.keys()) == [
        "aggr1",
        "aggr2",
        "is_valid_rt",
        "is_valid_t0",
        "is_valid_tmax",
    ]

    df = lh5.read_as("ch1084803/hit", outfile, "pd")

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


def test_build_hit_multiconfig(dsp_test_file, tmp_dir):
    out_file = f"{tmp_dir}/test_cal_geds_hit.lh5"

    # append the tmp_dir to the start of paths in the hit-multi-config.json
    with open(f"{config_dir}/hit-multi-config.json") as f:
        configdict = json.load(f)
    for key in configdict.keys():
        configdict[key] = f"{config_dir}/" + configdict[key].split("/")[-1]
    newdict = json.dumps(configdict)
    with open(f"{tmp_dir}/hit-multi-config.json", "w") as file:
        file.write(newdict)

    build_hit(
        dsp_test_file,
        outfile=out_file,
        lh5_tables_config=f"{tmp_dir}/hit-multi-config.json",
        wo_mode="of",
    )
    chans = ["ch1084803", "ch1084804", "ch1121600"]
    assert lh5.ls(out_file) == chans
    for ch in chans:
        assert lh5.ls(out_file, f"{ch}/") == [f"{ch}/hit"]
        assert set(lh5.ls(out_file, f"{ch}/hit/")) == {
            f"{ch}/hit/calE",
            f"{ch}/hit/AoE",
            f"{ch}/hit/A_max",
        }


def test_build_hit_calc(dsp_test_file, tmp_dir):
    out_file = f"{tmp_dir}/test_cal_geds_hit.lh5"

    build_hit(
        dsp_test_file,
        outfile=out_file,
        wo_mode="overwrite_file",
        lh5_tables_config=f"{config_dir}/hit-multi-config.json",
    )
    chans = ["ch1084803", "ch1084804", "ch1121600"]
    assert lh5.ls(out_file) == chans
    assert lh5.ls(out_file, "ch1084803/") == ["ch1084803/hit"]

    for ch in chans:
        df_hit = lh5.read_as(f"{ch}/hit/calE", out_file, "np")
        df_dsp = lh5.read_as(f"{ch}/dsp/trapEmax", dsp_test_file, "np")

        assert len(df_hit) == len(df_dsp)
        assert np.all(np.isclose(df_hit, np.sqrt(1.23 + 42.69 * (2 * df_dsp) ** 2)))


def test_vov_input(lgnd_test_data, tmp_dir):
    infile = lgnd_test_data.get_path(
        "lh5/l200-p03-r000-phy-20230312T055349Z-tier_psp.lh5"
    )
    outfile = f"{tmp_dir}/LDQTA_r117_20200110T105115Z_cal_geds_hit.lh5"

    hit_config = {
        "outputs": ["a"],
        "operations": {
            "a": {
                "expression": "a + m * energies",
                "parameters": {"a": 0, "m": 1},
            }
        },
    }

    build_hit(
        infile, outfile=outfile, hit_config=hit_config, wo_mode="of", buffer_len=9999999
    )

    orig = lh5.read_as("ch1067205/dsp/energies", infile, "ak")
    data = lh5.read_as("ch1067205/hit/a", outfile, "ak")
    assert ak.all(data == orig)
