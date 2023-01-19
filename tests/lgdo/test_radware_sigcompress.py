import numpy as np

from pygama.lgdo import LH5Store
from pygama.lgdo.compression import (
    _radware_sigcompress_decode,
    _radware_sigcompress_encode,
    radware_compress,
    radware_decompress,
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

    assert (enc_wf != wf).all()

    dec_wf = np.empty_like(wf)
    _radware_sigcompress_decode(enc_wf, dec_wf)

    assert (dec_wf == wf).all()

    comp_wf = radware_compress(wf)
    assert isinstance(comp_wf, np.ndarray)

    decomp_wf = radware_decompress(comp_wf)
    assert isinstance(decomp_wf, np.ndarray)

    assert (decomp_wf == wf).all()


def test_rawdware_sigcompress_performance(lgnd_test_data):
    store = LH5Store()
    obj, _ = store.read_object(
        "/geds/raw/waveform",
        lgnd_test_data.get_path("lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5"),
    )

    mean = 0
    for wf in obj["values"].nda:
        comp_wf = radware_compress(wf)
        mean += len(comp_wf) / len(wf)

    print(100 * (1 - mean / len(obj)))  # noqa: T201
