import os
from pathlib import Path

import pytest
from legend_testdata import LegendTestData

from pygama.dsp import build_dsp
from pygama.raw import build_raw

config_dir = Path(__file__).parent / "dsp" / "configs"


@pytest.fixture(scope="session")
def lgnd_test_data():
    ldata = LegendTestData()
    ldata.checkout("9c248f1")
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


@pytest.fixture(scope="session")
def multich_raw_file(lgnd_test_data):
    out_file = "/tmp/L200-comm-20211130-phy-spms.lh5"
    out_spec = {
        "FCEventDecoder": {
            "ch{key}": {
                "key_list": [[0, 6]],
                "out_stream": out_file + ":{name}",
                "out_name": "raw",
            }
        }
    }

    build_raw(
        in_stream=lgnd_test_data.get_path("fcio/L200-comm-20211130-phy-spms.fcio"),
        out_spec=out_spec,
        overwrite=True,
    )
    assert os.path.exists(out_file)

    return out_file


@pytest.fixture(scope="session")
def dsp_test_file_spm(multich_raw_file):
    chan_config = {
        "ch0/raw": f"{config_dir}/sipm-dsp-config.json",
        "ch1/raw": f"{config_dir}/sipm-dsp-config.json",
        "ch2/raw": f"{config_dir}/sipm-dsp-config.json",
    }

    out_file = "/tmp/L200-comm-20211130-phy-spms_dsp.lh5"
    build_dsp(
        multich_raw_file,
        out_file,
        {},
        n_max=5,
        lh5_tables=chan_config.keys(),
        chan_config=chan_config,
        write_mode="r",
    )

    assert os.path.exists(out_file)

    return out_file
