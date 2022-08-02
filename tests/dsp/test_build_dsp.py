import os
from pathlib import Path

import pytest

from pygama import lgdo
from pygama.dsp import build_dsp
from pygama.lgdo.lh5_store import LH5Store, ls
from pygama.raw import build_raw

config_dir = Path(__file__).parent / "configs"


@pytest.fixture(scope="module")
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

    return out_file


def test_build_dsp_basics(lgnd_test_data):
    build_dsp(
        lgnd_test_data.get_path("lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5"),
        "/tmp/LDQTA_r117_20200110T105115Z_cal_geds_dsp.lh5",
        dsp_config=f"{config_dir}/icpc-dsp-config.json",
        database={"pz": {"tau": 27460.5}},
        write_mode="r",
    )

    assert os.path.exists("/tmp/LDQTA_r117_20200110T105115Z_cal_geds_dsp.lh5")

    with pytest.raises(FileExistsError):
        build_dsp(
            lgnd_test_data.get_path("lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5"),
            "/tmp/LDQTA_r117_20200110T105115Z_cal_geds_dsp.lh5",
            dsp_config=f"{config_dir}/icpc-dsp-config.json",
        )

    with pytest.raises(FileNotFoundError):
        build_dsp(
            "non-existent-file.lh5",
            "/tmp/LDQTA_r117_20200110T105115Z_cal_geds_dsp.lh5",
            dsp_config=f"{config_dir}/icpc-dsp-config.json",
            write_mode="r",
        )


def test_build_dsp_channelwise(multich_raw_file):
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

    assert ls(out_file) == ["ch0", "ch1", "ch2", "dsp_info"]
    assert ls(out_file, "ch0/") == ["ch0/dsp"]
    assert ls(out_file, "ch0/dsp/") == ["ch0/dsp/bl_mean", "ch0/dsp/bl_std"]

    store = LH5Store()
    lh5_obj, n_rows = store.read_object("/ch0/dsp/bl_mean", out_file)
    assert isinstance(lh5_obj, lgdo.Array)
    assert len(lh5_obj) == 5
