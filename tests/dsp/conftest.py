import pytest

from pygama.lgdo import LH5Store
import pygama.dsp.processors  # noqa: F401


@pytest.fixture(scope='session')
def geds_raw_tbl(lgnd_test_data):
    store = LH5Store()
    obj, _ = store.read_object(
        '/geds/raw',
        lgnd_test_data.get_path('lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5'),
        n_rows=10)
    return obj
