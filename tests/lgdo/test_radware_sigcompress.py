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

    wf = obj["values"].nda[0].astype(np.uint16)

    enc_wf = np.empty_like(wf, dtype=np.uint16)
    _radware_sigcompress_encode(wf, enc_wf)

    assert enc_wf.dtype == np.uint16
    assert (enc_wf != wf).all()

    dec_wf = np.empty_like(wf, dtype=np.int16)
    _radware_sigcompress_decode(enc_wf, dec_wf)

    assert dec_wf.dtype == np.int16
    assert (dec_wf == wf).all()

    comp_wf = radware_compress(wf)
    assert isinstance(comp_wf, np.ndarray)
    assert comp_wf.dtype == np.uint16

    decomp_wf = radware_decompress(comp_wf)
    assert isinstance(decomp_wf, np.ndarray)
    assert decomp_wf.dtype == np.int16

    assert (decomp_wf == wf).all()


def test_rawdware_sigcompress_wftable(lgnd_test_data):
    store = LH5Store()
    wft, _ = store.read_object(
        "/geds/raw/waveform",
        lgnd_test_data.get_path("lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5"),
        n_rows=100,
    )

    enc_wft = radware_compress(wft)
    dec_wft = radware_decompress(enc_wft)

    assert dec_wft.t0 == wft.t0
    assert dec_wft.dt == wft.dt
    for wf1, wf2 in zip(dec_wft.values, wft.values):
        assert (wf1.astype(np.uint16) == wf2).all()


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
