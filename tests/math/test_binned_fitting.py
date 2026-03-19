from __future__ import annotations

import numpy as np
import pytest
from pytest import approx

import pygama.math.binned_fitting as pgbf


def test_fit_binned_and_goodness_of_fit():
    from numpy.random import normal

    from pygama.math.functions.gauss import nb_gauss_amp
    from pygama.math.histogram import get_hist

    np.random.seed(42)  # noqa: NPY002
    hist, bins, var = get_hist(normal(size=10000), bins=100, range=(-5, 5))  # noqa: NPY002
    fit, fit_error, fit_cov = pgbf.fit_binned(
        nb_gauss_amp, hist, bins, guess=(0, 0.9, 400), cost_func="Least Squares"
    )

    assert fit["mu"] == pytest.approx(-0.003933521046091256)
    assert fit["sigma"] == pytest.approx(0.9971419629418575)
    assert fit["a"] == pytest.approx(398.28437169188095)

    assert fit_error["mu"] == pytest.approx(0.010033194971234135)
    assert fit_error["sigma"] == pytest.approx(0.007291498082977738)
    assert fit_error["a"] == pytest.approx(4.930383892845187)

    assert np.allclose(
        fit_cov[0],
        np.array([[1.00665024e-04, -1.13519243e-06, 4.28011083e-04]], dtype=np.float32),
        rtol=1e-8,
    )

    chi, dof = pgbf.goodness_of_fit(
        hist, bins, var, nb_gauss_amp, fit, method="Pearson", scale_bins=True
    )
    assert chi == pytest.approx(82236.02419624529)
    assert dof == pytest.approx(97)

    poisson = pgbf.poisson_gof(fit, nb_gauss_amp, hist, bins, is_integral=False)

    assert poisson == pytest.approx(68204.56472918994)


def test_gauss_mode_width_max():
    from numpy.random import normal

    from pygama.math.histogram import get_hist

    np.random.seed(42)  # noqa: NPY002
    hist, bins, _var = get_hist(normal(size=10000), bins=100, range=(-5, 5))  # noqa: NPY002
    fit, _cov = pgbf.gauss_mode_width_max(hist, bins, n_bins=20)

    assert fit[0] == approx(0, abs=1e-1)
    assert fit[1] == approx(1, rel=1e-1)
    assert fit[2] == approx(398, abs=1)

    fit, _ = pgbf.gauss_mode_max(hist, bins)

    assert fit[0] == approx(0, abs=1e-1)
    assert fit[1] == approx(400, abs=1)

    fit, _err = pgbf.gauss_mode(hist, bins)

    assert fit == approx(0, abs=1e-1)


def test_gauss_mode_width_max_edge_cases():
    from pygama.math.histogram import get_hist

    np.random.seed(42)
    # histogram over range [-5, 5] with 100 bins, bin width = 0.1
    hist, bins, var = get_hist(
        np.random.normal(size=10000), bins=100, range=(-5, 5)
    )

    # mode_guess near the lower edge: find_bin returns a small index
    # so i_0 < floor(n_bins/2) triggers ValueError
    with pytest.raises(ValueError, match="Fit range exceeds histogram bounds"):
        pgbf.gauss_mode_width_max(hist, bins, mode_guess=-4.9, n_bins=5)

    # mode_guess near the upper edge: i_n = i_0 + n_bins >= len(hist)
    with pytest.raises(ValueError, match="Fit range exceeds histogram bounds"):
        pgbf.gauss_mode_width_max(hist, bins, mode_guess=4.9, n_bins=5)


def test_taylor_mode_max():
    from numpy.random import normal

    from pygama.math.histogram import get_hist

    np.random.seed(42)  # noqa: NPY002
    hist, bins, _var = get_hist(normal(size=10000), bins=100, range=(-5, 5))  # noqa: NPY002
    fit, _err = pgbf.taylor_mode_max(hist, bins, var=None, mode_guess=None, n_bins=5)

    assert fit[0] == pytest.approx(-0.005714285714285219)
    assert fit[1] == pytest.approx(400.4864285714285)
