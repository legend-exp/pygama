import os
from pathlib import Path

import pytest

from pygama.hit import build_hit
from pygama.lgdo import LH5Store, ls

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


def test_lh5_table_configs(dsp_test_file):
    outfile = "/tmp/LDQTA_r117_20200110T105115Z_cal_geds_hit.lh5"

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


def test_outputs_specification(dsp_test_file):
    outfile = "/tmp/LDQTA_r117_20200110T105115Z_cal_geds_hit.lh5"

    build_hit(
        dsp_test_file,
        outfile=outfile,
        hit_config=f"{config_dir}/basic-hit-config.json",
        wo_mode="overwrite",
    )

    store = LH5Store()
    obj, _ = store.read_object("/geds/hit", outfile)
    assert list(obj.keys()) == ["calE", "AoE", "A_max"]
