from pathlib import Path

import numpy as np
import pytest

from pygama.dsp.errors import DSPFatal
from pygama.dsp.processors import dplms


def test_dplms(compare_numba_vs_python):

    with open(Path(__file__).parent / "dplms_noise_mat.dat") as f:
        nmat = [[float(num) for num in line.split(" ")] for line in f]

    length = 50
    len_wf = 100
    ref = np.zeros(len_wf)
    ref[int(len_wf / 2 - 1) : int(len_wf / 2)] = 1

    # ensure the DSPFatal is raised for a negative length
    with pytest.raises(DSPFatal):
        dplms(nmat, ref, -1, 1, 1, 1, 1)

    # ensure the DSPFatal is raised for length not equal to noise matrix shape
    with pytest.raises(DSPFatal):
        dplms(nmat, ref, 10, 1, 1, 1, 1)

    # ensure the DSPFatal is raised for negative coefficients
    with pytest.raises(DSPFatal):
        dplms(nmat, ref, length, -1, 1, 1, 1)
    with pytest.raises(DSPFatal):
        dplms(nmat, ref, length, 1, -1, 1, 1)
    with pytest.raises(DSPFatal):
        dplms(nmat, ref, length, 1, 1, -1, 1)
    with pytest.raises(DSPFatal):
        dplms(nmat, ref, length, 1, 1, 1, -1)
    with pytest.raises(DSPFatal):
        dplms(nmat, ref, length, 1, 1, 1, 2)

    dplms_func = dplms(nmat, ref, length, 1, 1, 1, 1)

    # ensure to have a valid output
    w_in = np.ones(len_wf)
    w_out = np.empty(len_wf - length + 1)

    assert np.all(compare_numba_vs_python(dplms_func, w_in, w_out))

    # test if there is a nan in w_in
    w_in = np.ones(len_wf)
    w_in[4] = np.nan
    w_out = np.empty(len_wf - length + 1)

    assert np.all(compare_numba_vs_python(dplms_func, w_in, w_out))
