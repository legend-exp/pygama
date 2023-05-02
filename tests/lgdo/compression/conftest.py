import pytest

from pygama.lgdo import LH5Store


@pytest.fixture()
def wftable(lgnd_test_data):
    store = LH5Store()
    wft, _ = store.read_object(
        "/geds/raw/waveform",
        lgnd_test_data.get_path("lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5"),
    )
    return wft
