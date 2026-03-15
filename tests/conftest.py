from __future__ import annotations

from pathlib import Path

import pytest
from legendtestdata import LegendTestData

config_dir = Path(__file__).parent / "configs"


@pytest.fixture(scope="session")
def lgnd_test_data():
    ldata = LegendTestData()
    ldata.checkout("229cde0")
    return ldata


@pytest.fixture(scope="session")
def tmp_dir(tmpdir_factory):
    out_dir = tmpdir_factory.mktemp("data")
    assert Path(out_dir).exists()
    return out_dir


@pytest.fixture(scope="session")
def dsp_test_file(lgnd_test_data, tmp_dir):  # noqa: ARG001
    out_name = lgnd_test_data.get_path(
        "lh5/prod-ref-l200/generated/tier/dsp/cal/p03/r001/l200-p03-r001-cal-20230318T012144Z-tier_dsp.lh5"
    )
    assert Path(out_name).exists()

    return out_name


@pytest.fixture(scope="session")
def raw_test_file(lgnd_test_data, tmp_dir):  # noqa: ARG001

    out_name = lgnd_test_data.get_path(
        "lh5/prod-ref-l200/generated/tier/raw/phy/p03/r001/l200-p03-r001-phy-20230322T160139Z-tier_raw.lh5"
    )
    assert Path(out_name).exists()

    return out_name
