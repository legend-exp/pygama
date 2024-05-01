import numpy as np
from pytest import approx

import pygama.math.binned_fitting as pgbf


def test_fit_binned_and_goodness_of_fit():
    from numpy.random import normal

    from pygama.math.functions.gauss import nb_gauss_amp
    from pygama.math.histogram import get_hist

    np.random.seed(42)
    hist, bins, var = get_hist(normal(size=10000), bins=100, range=(-5, 5))
    fit, fit_error, fit_cov = pgbf.fit_binned(
        nb_gauss_amp, hist, bins, guess=(0, 0.9, 400), cost_func="Least Squares"
    )

    assert fit["mu"] == approx(-0.003933521046091256)
    assert fit["sigma"] == approx(0.9971419629418575)
    assert fit["a"] == approx(398.28437169188095)

    assert fit_error["mu"] == approx(0.010033194971234135)
    assert fit_error["sigma"] == approx(0.007291498082977738)
    assert fit_error["a"] == approx(4.930383892845187)

    assert np.allclose(
        fit_cov[0],
        np.array([[1.00665024e-04, -1.13519243e-06, 4.28011083e-04]], dtype=np.float32),
        rtol=1e-8,
    )

    chi, dof = pgbf.goodness_of_fit(
        hist, bins, var, nb_gauss_amp, fit, method="Pearson", scale_bins=True
    )
    assert chi == approx(82236.02419624529)
    assert dof == approx(97)

    poisson = pgbf.poisson_gof(fit, nb_gauss_amp, hist, bins, is_integral=False)

    assert poisson == approx(68204.56472918994)


def test_gauss_mode_width_max():
    from numpy.random import normal

    from pygama.math.histogram import get_hist

    np.random.seed(42)
    hist, bins, var = get_hist(normal(size=10000), bins=100, range=(-5, 5))
    fit, cov = pgbf.gauss_mode_width_max(hist, bins, n_bins=20)

    assert fit[0] == approx(-0.006127842485254326)
    assert fit[1] == approx(1.0066159284176235)
    assert fit[2] == approx(398.35078610804527)

    fit, err = pgbf.gauss_mode_max(hist, bins)

    assert fit[0] == approx(-0.0028635117051317638)
    assert fit[1] == approx(400.2130565573762)

    fit, err = pgbf.gauss_mode(hist, bins)

    assert fit == approx(-0.0028635117051317638)


def test_taylor_mode_max():
    from numpy.random import normal

    from pygama.math.histogram import get_hist

    np.random.seed(42)
    hist, bins, var = get_hist(normal(size=10000), bins=100, range=(-5, 5))
    fit, err = pgbf.taylor_mode_max(hist, bins, var=None, mode_guess=None, n_bins=5)

    assert fit[0] == approx(-0.005714285714285219)
    assert fit[1] == approx(400.4864285714285)
