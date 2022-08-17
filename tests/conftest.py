import os
from pathlib import Path

import pytest
from legend_testdata import LegendTestData

from pygama.dsp import build_dsp

config_dir = Path(__file__).parent / "dsp" / "configs"


@pytest.fixture(scope="session")
def lgnd_test_data():
    ldata = LegendTestData()
    ldata.checkout("968c9ba")
    return ldata


@pytest.fixture(scope="session")
def dsp_test_file(lgnd_test_data):
    out_name = "/tmp/LDQTA_r117_20200110T105115Z_cal_geds_dsp.lh5"
    build_dsp(
        lgnd_test_data.get_path("lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5"),
        out_name,
        dsp_config=f"{config_dir}/icpc-dsp-config.json",
        database={"pz": {"tau": 27460.5}},
        write_mode="r",
    )
    assert os.path.exists(out_name)

    return out_name
