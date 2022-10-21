import numpy as np
import pytest

from pygama.dsp.errors import DSPFatal
from pygama.dsp.processors import fixed_time_pickoff


def test_fixed_time_pickoff():
    """Testing function for the fixed_time_pickoff processor."""

    len_wf = 20

    w_in = np.ones(len_wf)
    w_in[4] = np.nan
    assert np.isnan(fixed_time_pickoff(w_in, 1, ord("i")))

    w_in = np.ones(len_wf)
    assert np.isnan(fixed_time_pickoff(w_in, np.nan, ord("i")))

    w_in = np.ones(len_wf)
    assert np.isnan(fixed_time_pickoff(w_in, -1, ord("i")))

    w_in = np.ones(len_wf)
    assert np.isnan(fixed_time_pickoff(w_in, len_wf, ord("i")))

    with pytest.raises(DSPFatal):
        w_in = np.ones(len_wf)
        fixed_time_pickoff(w_in, 1.5, ord("i"))

    with pytest.raises(DSPFatal):
        w_in = np.ones(len_wf)
        fixed_time_pickoff(w_in, 1.5, ord(" "))

    # linear tests
    w_in = np.arange(len_wf, dtype=float)
    assert fixed_time_pickoff(w_in, 3, ord("i")) == 3

    chars = ["n", "f", "c", "l", "h", "s"]
    sols = [4, 3, 4, 3.5, 3.5, 3.5]

    for char, sol in zip(chars, sols):
        assert fixed_time_pickoff(w_in, 3.5, ord(char)) == sol

    # sine wave tests
    w_in = np.sin(np.arange(len_wf))

    chars = ["n", "f", "c", "l", "h", "s"]
    sols = [
        0.1411200080598672,
        0.1411200080598672,
        -0.7568024953079282,
        -0.08336061778208165,
        -0.09054574599004982,
        -0.10707938709427486,
    ]

    for char, sol in zip(chars, sols):
        assert np.isclose(fixed_time_pickoff(w_in, 3.25, ord(char)), sol)

    # last few corner cases of 'h'
    w_in = np.sin(np.arange(len_wf))
    ftps = [0.2, len_wf - 1.8]
    sols = [
        0.1806725096462211,
        -0.6150034250096629,
    ]

    for ftp, sol in zip(ftps, sols):
        assert np.isclose(fixed_time_pickoff(w_in, ftp, ord("h")), sol)
