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


def test_rawdware_sigcompress_special_cases():
    # fmt: off
    wf = np.array([5, -7, -14, 17, -21, 0, -10, -2, -17, -14, 22, -5, -7, 7,
                   14, -2, 1, 0, -7, -21, -5, -15, -5, 11, 2, -24, 18, 2, -9,
                   -3, 4, 7, -14, 12, 8, -23, 9, -8, 6, -2, 3, 9, -4, -10, -20,
                   5, 9, 12, 21, 3, 95, 189, 294, 405, 487, 593, 702, 781, 896,
                   1034, 986, 1004, 1011, 996, 990, 1005, 1003, 987, 1003, 994,
                   1008, 996, 989, 995, 997, 996, 985, 1010, 1006, 995, 988,
                   993, 1013, 1005, 985, 1025, 1008, 995, 1004, 1003])

    enc_wf_exp = np.array([90, 50, 6, 65512, 29970, 43277, 33686, 7339, 37701,
                           63894, 25988, 17228, 38115, 26634, 39485, 22303,
                           10824, 389, 1942, 28181, 3601, 55396, 46512, 40,
                           40, 95, 65488, 36505, 40834, 39581, 32675, 47616,
                           16951, 8490, 16174, 8256, 10046, 9257, 13874,
                           12069, 18732, 9513, 13636, 10268, 22559, 9017,
                           12032, 0])

    assert (radware_compress(wf) == enc_wf_exp).all()

    wf = np.array([107, 105, 113, 112, 105, 91, 119, 126, 110, 117, 105, 98,
                   129, 91, 112, 102, -33, 213, -54, 312, 107, 97, 107, 123,
                   114, 88, 130, 114, 103, 109, 116, 119, 98, 124, 120, 89,
                   121, 104, 118, 110, 115, 121, 108, 102, 92, 117, 121, 124,
                   133, 115, 207, 301, 406, 517, 599, 705, 814, 893, 1008,
                   1146, 1098, 1116, 1123, 1108, 1102, 1117, 1115, 1099, 1115,
                   1106, 1120, 1108, 1101, 1107, 1109, 1108, 1097, 1122, 1118,
                   1107, 1100, 1105, 1125, 1117, 1097, 1137, 1120, 1107, 1116,
                   1115])

    enc_wf_exp = np.array([90, 53, 9, 65482, 20647, 54506, 25850, 17754, 46162,
                           10963, 59781, 47685, 19612, 2754, 49174, 58634,
                           23874, 45396, 9111, 2692, 60045, 21677, 19500,
                           38344, 62842, 31064, 42068, 43988, 18884, 37549,
                           24242, 23978, 24758, 15968, 37, 40, 517, 65488,
                           33434, 40319, 41914, 66, 14113, 10815, 11808, 16423,
                           15908, 10550, 12847, 9545, 11301, 10549, 17448,
                           7256, 7971, 14639])

    assert (radware_compress(wf) == enc_wf_exp).all()
    # fmt: on


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
