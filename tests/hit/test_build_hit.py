import pytest

import os
from pathlib import Path

from pygama.hit import build_hit
from pygama.lgdo import ls

config_dir = Path(__file__).parent / "configs"


def test_basics(dsp_test_file):
    outfile = "/tmp/LDQTA_r117_20200110T105115Z_cal_geds_hit.lh5"

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


def test_build_hit_table_configs(dsp_test_file):
    outfile = "/tmp/LDQTA_r117_20200110T105115Z_cal_geds_hit.lh5"

    lh5_tables_config = {
        "/geds/dsp": f"{config_dir}/basic-hit-config.json"
    }

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
            "calE": {
                "expression": "sqrt(@a + @b * trapEmax**2)",
                "parameters": {"a": "1.23", "b": "42.69"}
            },
            "AoE": {"expression": "A_max/calE"}
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
