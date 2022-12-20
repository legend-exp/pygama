import numpy as np
import pytest

from pygama.dsp.errors import DSPFatal
from pygama.dsp.processors import discrete_wavelet_transform


def test_discrete_wavelet_transform(compare_numba_vs_python):
    """Testing function for the discrete_wavelet_transform processor."""

    # set up values to use for each test case
    len_wf_in = 16
    wave_type = "haar"
    level = 2
    len_wf_out = 4

    # ensure the DSPFatal is raised for a negative level
    with pytest.raises(DSPFatal):
        discrete_wavelet_transform(wave_type, -1)

    # ensure that a valid input gives the expected output
    w_in = np.ones(len_wf_in)
    w_out = np.empty(len_wf_out)
    w_out_expected = np.ones(len_wf_out) * 2 ** (level / 2)

    dwt_func = discrete_wavelet_transform(wave_type, level)
    assert np.allclose(
        compare_numba_vs_python(dwt_func, w_in, w_out),
        w_out_expected,
    )

    # ensure that if there is a nan in w_in, all nans are outputted
    w_in = np.ones(len_wf_in)
    w_in[4] = np.nan
    w_out = np.empty(len_wf_out)

    dwt_func = discrete_wavelet_transform(wave_type, level)
    assert np.all(np.isnan(compare_numba_vs_python(dwt_func, w_in, w_out)))
