import inspect

import numpy as np
import pytest

from pygama.dsp.errors import DSPFatal
from pygama.dsp.processors import get_multi_local_extrema

# symmetric delta tests (all searches should behave identical)


# test with symmetric delta L->R (dont care about abs. thresholds here)
def test_get_multi_local_extrema_ltor(compare_numba_vs_python):
    wf = np.array(
        [
            0,
            0,
            1,
            2,
            3,
            4,
            5,
            4,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            9,
            8,
            7,
            6,
            5,
            4,
            3,
            4,
            5,
            4,
            3,
            2,
            1,
            0,
            0,
        ]
    )
    max_out = np.zeros(3)
    max_out[:] = np.nan
    min_out = np.zeros(3)
    min_out[:] = np.nan
    n_min_out = np.zeros(1)
    n_max_out = np.zeros(1)
    flout = np.zeros(1)

    compare_numba_vs_python(
        get_multi_local_extrema,
        wf,
        3,
        3,
        0,
        0,
        20,
        max_out,
        min_out,
        n_max_out,
        n_min_out,
        flout,
    )

    assert np.array_equal(max_out, np.array([15, np.nan, np.nan]), equal_nan=True)
    assert np.array_equal(min_out, np.array([np.nan, np.nan, np.nan]), equal_nan=True)
    assert n_max_out[0] == 1
    assert flout[0] == 1
    assert n_min_out[0] == 0


# test with symmetric delta L<-R (dont care about abs. thresholds here)
def test_get_multi_local_extrema_rtol(compare_numba_vs_python):
    wf = np.array(
        [
            0,
            0,
            1,
            2,
            3,
            4,
            5,
            4,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            9,
            8,
            7,
            6,
            5,
            4,
            3,
            4,
            5,
            4,
            3,
            2,
            1,
            0,
            0,
        ]
    )
    max_out = np.zeros(3)
    max_out[:] = np.nan
    min_out = np.zeros(3)
    min_out[:] = np.nan
    n_min_out = np.zeros(1)
    n_max_out = np.zeros(1)
    flout = np.zeros(1)

    compare_numba_vs_python(
        get_multi_local_extrema,
        wf,
        3,
        3,
        1,
        0,
        20,
        max_out,
        min_out,
        n_max_out,
        n_min_out,
        flout,
    )

    assert np.array_equal(max_out, np.array([15, np.nan, np.nan]), equal_nan=True)
    assert np.array_equal(min_out, np.array([np.nan, np.nan, np.nan]), equal_nan=True)
    assert n_max_out[0] == 1
    assert flout[0] == 1
    assert n_min_out[0] == 0


# test with symmetric delta L<-c->R (dont care about abs. thresholds here)
def test_get_multi_local_extrema_both_cons(compare_numba_vs_python):
    wf = np.array(
        [
            0,
            0,
            1,
            2,
            3,
            4,
            5,
            4,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            9,
            8,
            7,
            6,
            5,
            4,
            3,
            4,
            5,
            4,
            3,
            2,
            1,
            0,
            0,
        ]
    )
    max_out = np.zeros(3)
    max_out[:] = np.nan
    min_out = np.zeros(3)
    min_out[:] = np.nan
    n_min_out = np.zeros(1)
    n_max_out = np.zeros(1)
    flout = np.zeros(1)

    compare_numba_vs_python(
        get_multi_local_extrema,
        wf,
        3,
        3,
        2,
        0,
        20,
        max_out,
        min_out,
        n_max_out,
        n_min_out,
        flout,
    )

    assert np.array_equal(max_out, np.array([15, np.nan, np.nan]), equal_nan=True)
    assert np.array_equal(min_out, np.array([np.nan, np.nan, np.nan]), equal_nan=True)
    assert n_max_out[0] == 1
    assert flout[0] == 1
    assert n_min_out[0] == 0


# test with symmetric delta L<-a->R (dont care about abs. thresholds here)
def test_get_multi_local_extrema_both_agro(compare_numba_vs_python):
    wf = np.array(
        [
            0,
            0,
            1,
            2,
            3,
            4,
            5,
            4,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            9,
            8,
            7,
            6,
            5,
            4,
            3,
            4,
            5,
            4,
            3,
            2,
            1,
            0,
            0,
        ]
    )
    max_out = np.zeros(3)
    max_out[:] = np.nan
    min_out = np.zeros(3)
    min_out[:] = np.nan
    n_min_out = np.zeros(1)
    n_max_out = np.zeros(1)
    flout = np.zeros(1)

    compare_numba_vs_python(
        get_multi_local_extrema,
        wf,
        3,
        3,
        3,
        0,
        20,
        max_out,
        min_out,
        n_max_out,
        n_min_out,
        flout,
    )

    assert np.array_equal(max_out, np.array([15, np.nan, np.nan]), equal_nan=True)
    assert np.array_equal(min_out, np.array([np.nan, np.nan, np.nan]), equal_nan=True)
    assert n_max_out[0] == 1
    assert flout[0] == 1
    assert n_min_out[0] == 0


# asymmetric delta test (result should be search direction dependent)

# test with asymmetric delta L->R (dont care about abs. thresholds here)
def test_get_multi_local_extrema_ltor_asym(compare_numba_vs_python):
    wf = np.array(
        [
            0,
            0,
            1,
            2,
            3,
            4,
            5,
            4,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            9,
            8,
            7,
            6,
            5,
            4,
            3,
            4,
            5,
            4,
            3,
            2,
            1,
            0,
            0,
        ]
    )
    max_out = np.zeros(3)
    max_out[:] = np.nan
    min_out = np.zeros(3)
    min_out[:] = np.nan
    n_min_out = np.zeros(1)
    n_max_out = np.zeros(1)
    flout = np.zeros(1)

    compare_numba_vs_python(
        get_multi_local_extrema,
        wf,
        3,
        1,
        0,
        0,
        20,
        max_out,
        min_out,
        n_max_out,
        n_min_out,
        flout,
    )

    assert np.array_equal(max_out, np.array([15, 24, np.nan]), equal_nan=True)
    assert np.array_equal(min_out, np.array([22, np.nan, np.nan]), equal_nan=True)
    assert n_max_out[0] == 2
    assert flout[0] == 0
    assert n_min_out[0] == 1


# test with asymmetric delta L<-R (dont care about abs. thresholds here)
def test_get_multi_local_extrema_rtol_asym(compare_numba_vs_python):
    wf = np.array(
        [
            0,
            0,
            1,
            2,
            3,
            4,
            5,
            4,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            9,
            8,
            7,
            6,
            5,
            4,
            3,
            4,
            5,
            4,
            3,
            2,
            1,
            0,
            0,
        ]
    )
    max_out = np.zeros(3)
    max_out[:] = np.nan
    min_out = np.zeros(3)
    min_out[:] = np.nan
    n_min_out = np.zeros(1)
    n_max_out = np.zeros(1)
    flout = np.zeros(1)

    compare_numba_vs_python(
        get_multi_local_extrema,
        wf,
        3,
        1,
        1,
        0,
        20,
        max_out,
        min_out,
        n_max_out,
        n_min_out,
        flout,
    )

    assert np.array_equal(max_out, np.array([15, 6, np.nan]), equal_nan=True)
    assert np.array_equal(min_out, np.array([8, np.nan, np.nan]), equal_nan=True)
    assert n_max_out[0] == 2
    assert flout[0] == 0
    assert n_min_out[0] == 1


# test with asymmetric delta L<-c->R (dont care about abs. thresholds here)
def test_get_multi_local_extrema_both_cons_asym(compare_numba_vs_python):
    wf = np.array(
        [
            0,
            0,
            1,
            2,
            3,
            4,
            5,
            4,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            9,
            8,
            7,
            6,
            5,
            4,
            3,
            4,
            5,
            4,
            3,
            2,
            1,
            0,
            0,
        ]
    )
    max_out = np.zeros(3)
    max_out[:] = np.nan
    min_out = np.zeros(3)
    min_out[:] = np.nan
    n_min_out = np.zeros(1)
    n_max_out = np.zeros(1)
    flout = np.zeros(1)

    compare_numba_vs_python(
        get_multi_local_extrema,
        wf,
        3,
        1,
        2,
        0,
        20,
        max_out,
        min_out,
        n_max_out,
        n_min_out,
        flout,
    )

    assert np.array_equal(max_out, np.array([15, np.nan, np.nan]), equal_nan=True)
    assert np.array_equal(min_out, np.array([np.nan, np.nan, np.nan]), equal_nan=True)
    assert n_max_out[0] == 1
    assert flout[0] == 1
    assert n_min_out[0] == 0


# test with asymmetric delta L<-a->R (dont care about abs. thresholds here)
def test_get_multi_local_extrema_both_agro_asym(compare_numba_vs_python):
    wf = np.array(
        [
            0,
            0,
            1,
            2,
            3,
            4,
            5,
            4,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            9,
            8,
            7,
            6,
            5,
            4,
            3,
            4,
            5,
            4,
            3,
            2,
            1,
            0,
            0,
        ]
    )
    max_out = np.zeros(3)
    max_out[:] = np.nan
    min_out = np.zeros(3)
    min_out[:] = np.nan
    n_min_out = np.zeros(1)
    n_max_out = np.zeros(1)
    flout = np.zeros(1)

    compare_numba_vs_python(
        get_multi_local_extrema,
        wf,
        3,
        1,
        3,
        0,
        20,
        max_out,
        min_out,
        n_max_out,
        n_min_out,
        flout,
    )

    assert np.array_equal(max_out, np.array([6, 15, 24]), equal_nan=True)
    assert np.array_equal(min_out, np.array([8, 22, np.nan]), equal_nan=True)
    assert n_max_out[0] == 3
    assert flout[0] == 0
    assert n_min_out[0] == 2


# Absolute threshold test with aggressive both direction search
def test_get_multi_local_extrema_both_agro_asym_abs(compare_numba_vs_python):
    wf = np.array(
        [
            0,
            0,
            1,
            2,
            3,
            4,
            5,
            4,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            9,
            8,
            7,
            6,
            5,
            4,
            3,
            4,
            5,
            4,
            3,
            2,
            1,
            0,
            0,
        ]
    )
    max_out = np.zeros(3)
    max_out[:] = np.nan
    min_out = np.zeros(3)
    min_out[:] = np.nan
    n_min_out = np.zeros(1)
    n_max_out = np.zeros(1)
    flout = np.zeros(1)

    compare_numba_vs_python(
        get_multi_local_extrema,
        wf,
        3,
        1,
        3,
        8,
        20,
        max_out,
        min_out,
        n_max_out,
        n_min_out,
        flout,
    )

    assert np.array_equal(max_out, np.array([15, np.nan, np.nan]), equal_nan=True)
    assert np.array_equal(min_out, np.array([8, 22, np.nan]), equal_nan=True)
    assert n_max_out[0] == 1
    assert flout[0] == 1
    assert n_min_out[0] == 2


# And now test if everything works well when vectorized

# return nan tests
def test_get_multi_local_extrema_return_on_nan(compare_numba_vs_python):
    wf = np.ones(20)
    max_out = np.zeros(3)
    max_out[:] = np.nan
    min_out = np.zeros(3)
    min_out[:] = np.nan
    n_min_out = np.zeros(1)
    n_max_out = np.zeros(1)
    flout = np.zeros(1)

    wf[4] = np.nan
    assert np.isnan(
        compare_numba_vs_python(
            get_multi_local_extrema,
            wf,
            3,
            1,
            3,
            8,
            20,
            max_out,
            min_out,
            n_max_out,
            n_min_out,
            flout,
        )
    )

    wf[4] = 3
    assert np.isnan(
        compare_numba_vs_python(
            get_multi_local_extrema,
            wf,
            np.nan,
            3,
            3,
            8,
            20,
            max_out,
            min_out,
            n_max_out,
            n_min_out,
            flout,
        )
    )
    assert np.isnan(
        compare_numba_vs_python(
            get_multi_local_extrema,
            wf,
            np.nan,
            3,
            np.nan,
            8,
            20,
            max_out,
            min_out,
            n_max_out,
            n_min_out,
            flout,
        )
    )
    assert np.isnan(
        compare_numba_vs_python(
            get_multi_local_extrema,
            wf,
            np.nan,
            np.nan,
            3,
            8,
            20,
            max_out,
            min_out,
            n_max_out,
            n_min_out,
            flout,
        )
    )


# DSP Fatal test
def test_get_multi_local_extrema_dsp_fatal():
    wf = np.ones(20)
    max_out = np.zeros(len(wf) + 1)
    max_out[:] = np.nan
    min_out = np.zeros(len(wf) + 1)
    min_out[:] = np.nan
    n_min_out = np.zeros(1)
    n_max_out = np.zeros(1)
    flout = np.zeros(1)

    with pytest.raises(DSPFatal):
        get_multi_local_extrema(
            wf, 3, 1, 3, 8, 20, max_out, min_out, n_max_out, n_min_out, flout
        )
    with pytest.raises(DSPFatal):
        inspect.unwrap(get_multi_local_extrema)(
            wf, 3, 1, 3, 8, 20, max_out, min_out, n_max_out, n_min_out, flout
        )

    max_out = np.zeros(3)
    max_out[:] = np.nan
    min_out = np.zeros(3)
    min_out[:] = np.nan

    with pytest.raises(DSPFatal):
        get_multi_local_extrema(
            wf, -3, 1, 3, 8, 20, max_out, min_out, n_max_out, n_min_out, flout
        )

    with pytest.raises(DSPFatal):
        inspect.unwrap(get_multi_local_extrema)(
            wf, -3, 1, 3, 8, 20, max_out, min_out, n_max_out, n_min_out, flout
        )

    with pytest.raises(DSPFatal):
        get_multi_local_extrema(
            wf, 3, -1, 3, 8, 20, max_out, min_out, n_max_out, n_min_out, flout
        )

    with pytest.raises(DSPFatal):
        inspect.unwrap(get_multi_local_extrema)(
            wf, 3, -1, 3, 8, 20, max_out, min_out, n_max_out, n_min_out, flout
        )

    with pytest.raises(DSPFatal):
        get_multi_local_extrema(
            wf, 3, 1, -3, 8, 20, max_out, min_out, n_max_out, n_min_out, flout
        )

    with pytest.raises(DSPFatal):
        inspect.unwrap(get_multi_local_extrema)(
            wf, 3, 1, -3, 8, 20, max_out, min_out, n_max_out, n_min_out, flout
        )
