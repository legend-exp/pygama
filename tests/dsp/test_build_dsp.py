from pathlib import Path

import pytest

from pygama import lgdo
from pygama.dsp import build_dsp
from pygama.lgdo.lh5_store import LH5Store, ls

config_dir = Path(__file__).parent / "configs"


def test_build_dsp_basics(lgnd_test_data, dsp_test_file):
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


def test_build_dsp_spms_channelwise(dsp_test_file_spm):

    assert ls(dsp_test_file_spm) == ["ch0", "ch1", "ch2", "dsp_info"]
    assert ls(dsp_test_file_spm, "ch0/") == ["ch0/dsp"]
    assert ls(dsp_test_file_spm, "ch0/dsp/") == [
        "ch0/dsp/energies",
        "ch0/dsp/trigger_pos",
    ]

    store = LH5Store()
    lh5_obj, n_rows = store.read_object("/ch0/dsp/energies", dsp_test_file_spm)
    assert isinstance(lh5_obj, lgdo.ArrayOfEqualSizedArrays)
    assert len(lh5_obj) == 5
    assert len(lh5_obj.nda[0]) == 20
