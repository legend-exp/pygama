import numpy as np

from pygama.lgdo import LH5Store
from pygama.lgdo.compression import (
    _radware_sigcompress_decode,
    _radware_sigcompress_encode,
)


def test_rawdware_sigcompress(lgnd_test_data):
    store = LH5Store()
    obj, _ = store.read_object(
        "/geds/raw/waveform",
        lgnd_test_data.get_path("lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5"),
        n_rows=1,
    )

    wf = obj["values"].nda[0]

    enc_wf = np.empty_like(wf)
    _radware_sigcompress_encode(wf, enc_wf)

    dec_wf = np.empty_like(wf)
    _radware_sigcompress_decode(enc_wf, dec_wf)

    assert (dec_wf == wf).all()
