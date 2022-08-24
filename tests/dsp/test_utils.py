from pygama.dsp.utils import numba_defaults


def test_numba_defaults_loading():
    numba_defaults.cache = False
    numba_defaults.boundscheck = True
