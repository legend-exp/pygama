import os
from pathlib import Path

import pytest
from dspeed import build_dsp
from legendtestdata import LegendTestData

config_dir = Path(__file__).parent / "configs"


@pytest.fixture(scope="session")
def lgnd_test_data():
    ldata = LegendTestData()
    ldata.checkout("756aef8")
    return ldata


@pytest.fixture(scope="session")
def tmp_dir(tmpdir_factory):
    out_dir = tmpdir_factory.mktemp("data")
    assert os.path.exists(out_dir)
    return out_dir


@pytest.fixture(scope="session")
def dsp_test_file(lgnd_test_data, tmp_dir):
    out_name = f"{tmp_dir}/LDQTA_r117_20200110T105115Z_cal_geds_dsp.lh5"
    build_dsp(
        lgnd_test_data.get_path("lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5"),
        out_name,
        dsp_config=f"{config_dir}/icpc-dsp-config.json",
        database={"pz": {"tau": 27460.5}},
        write_mode="r",
    )
    assert os.path.exists(out_name)

    return out_name


@pytest.fixture(scope="session")
def dsp_test_file_spm(lgnd_test_data, tmp_dir):
    chan_config = {
        "ch0/raw": f"{config_dir}/sipm-dsp-config.json",
        "ch1/raw": f"{config_dir}/sipm-dsp-config.json",
        "ch2/raw": f"{config_dir}/sipm-dsp-config.json",
    }

    out_file = f"{tmp_dir}/L200-comm-20211130-phy-spms_dsp.lh5"
    build_dsp(
        lgnd_test_data.get_path("lh5/L200-comm-20211130-phy-spms.lh5"),
        out_file,
        {},
        n_max=5,
        lh5_tables=chan_config.keys(),
        chan_config=chan_config,
        write_mode="r",
    )

    assert os.path.exists(out_file)

    return out_file
