import numpy as np
import pytest

import pygama.lgdo as lgdo
from pygama.lgdo.lh5_store import LH5Iterator


@pytest.fixture(scope="module")
def lgnd_file(lgnd_test_data):
    return lgnd_test_data.get_path("lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5")


def test_basics(lgnd_file):
    lh5_it = LH5Iterator(
        lgnd_file,
        "/geds/raw",
        entry_list=range(100),
        field_mask=["baseline"],
        buffer_len=5,
    )

    lh5_obj, n_rows = lh5_it.read(4)
    assert n_rows == 5
    assert isinstance(lh5_obj, lgdo.Table)
    assert list(lh5_obj.keys()) == ["baseline"]
    assert (
        lh5_obj["baseline"].nda == np.array([14353, 14254, 14525, 11656, 13576])
    ).all()

    for lh5_obj, entry, n_rows in lh5_it:
        assert len(lh5_obj) == 5
        assert n_rows == 5
        assert entry % 5 == 0


def test_errors(lgnd_file):
    with pytest.raises(RuntimeError):
        LH5Iterator("non-existent-file.lh5", "random-group")

    with pytest.raises(ValueError):
        LH5Iterator(1, 2)


def test_lgnd_waveform_table_fancy_idx(lgnd_file):
    lh5_it = LH5Iterator(
        lgnd_file,
        "geds/raw/waveform",
        entry_list=[
            7,
            9,
            25,
            27,
            33,
            38,
            46,
            52,
            57,
            59,
            67,
            71,
            72,
            82,
            90,
            92,
            93,
            94,
            97,
        ],
        buffer_len=5,
    )

    lh5_obj, n_rows = lh5_it.read(0)
    assert isinstance(lh5_obj, lgdo.WaveformTable)
    assert len(lh5_obj) == 5
