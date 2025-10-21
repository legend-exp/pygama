import os
from pathlib import Path

import pytest
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
    out_name = lgnd_test_data.get_path(
        "lh5/prod-ref-l200/generated/tier/dsp/cal/p03/r001/l200-p03-r001-cal-20230318T012144Z-tier_dsp.lh5"
    )
    assert os.path.exists(out_name)

    return out_name
