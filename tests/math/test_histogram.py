import numpy as np
from pytest import approx

import pygama.math.histogram as pgh


def test_get_hist():
    data = np.arange(-10, 10)
    hist, bins, err = pgh.get_hist(data, bins=20, range=(-10, 10))
    numpy_hist, numpy_bins = np.histogram(data, bins=20, range=(-10, 10))
    # Check that we agree with numpy
    assert hist == approx(numpy_hist, rel=1e-5)
    assert bins == approx(numpy_bins, rel=1e-5)
    assert err == approx(
        numpy_hist, rel=1e-5
    )  # for no weights, the variances should be the same as the histogram!
    # check that the values match what we expect
    assert np.array_equal(np.array(hist, dtype=np.float32), np.ones(len(data)))
    assert np.array_equal(np.array(bins, dtype=np.float32), np.arange(-10, 11))
    assert np.array_equal(np.array(err, dtype=np.float32), np.ones(len(data)))

    weight = np.full(len(data), 0.2)

    hist, bins, var = pgh.get_hist(data, bins=50, range=(-10, 10), dx=2, wts=weight)
    numpy_hist, numpy_bins = np.histogram(
        data, bins=10, range=(-10, 10), weights=weight
    )
    numpy_var, numpy_bins = np.histogram(
        data, bins=10, range=(-10, 10), weights=weight * weight
    )

    assert hist == approx(numpy_hist, rel=1e-5)
    assert bins == approx(numpy_bins, rel=1e-5)
    assert var == approx(numpy_var, rel=1e-5)

    assert np.allclose(
        hist, np.full(10, 2 * 0.2), rtol=1e-8
    )  # each bin has two elements, and each element has the 0.2 weight
    assert np.allclose(
        bins, np.array(np.arange(-10, 11, 2), dtype=np.float64), rtol=1e-8
    )
    assert np.allclose(var, np.full(10, 2 * 0.2 * 0.2), rtol=1e-8)


def test_better_int_binning():
    lo, hi, bins = pgh.better_int_binning(x_lo=-20.1, x_hi=10.1, n_bins=10.2)
    assert lo == -21
    assert hi == 12
    assert bins == 11


def test_get_bin_centers():
    bins = pgh.get_bin_centers(np.arange(-10.5, 10))
    assert np.array_equal(bins, np.arange(-10, 10))


def test_get_bin_widths():
    dx = pgh.get_bin_widths(np.arange(-10, 10))
    assert np.array_equal(dx, np.ones(len(dx)))


def test_find_bin():
    uniform_bins = np.arange(-10, 10)
    uniform_idx = pgh.find_bin(1.5, uniform_bins)
    assert uniform_idx == 11

    non_uniform_bins = np.array(
        [-1, -0.6, -0.3, 0, 0.1, 0.5, 0.76, 1, 1.1, 1.4, 1.6, 2], dtype=np.float32
    )
    non_uniform_idx = pgh.find_bin(1.5, non_uniform_bins)
    assert non_uniform_idx == 10


def test_range_slice():
    np.random.seed(42)
    data = np.random.normal(size=100).astype(np.float32)
    hist, bins, err = pgh.get_hist(data, bins=10, range=(-1, 1))

    hist_slice, bins_slice, var_slice = pgh.range_slice(0.1, 0.6, hist, bins, err)
    assert np.array_equal(hist_slice, [8.0, 13.0, 3.0])
    assert np.array_equal(
        np.array(bins_slice, dtype=np.float32),
        np.array([0.0, 0.2, 0.4, 0.6], dtype=np.float32),
    )
    assert np.array_equal(var_slice, [8.0, 13.0, 3.0])


def test_get_fwfm():
    from numpy.random import normal

    np.random.seed(42)
    hist, bins, var = pgh.get_hist(normal(size=10000), bins=100, range=(-5, 5))

    fwfm, dfwfm = pgh.get_fwfm(0.5, hist, bins, var, method="bins_over_f")
    assert fwfm == approx(2.4)
    assert dfwfm == approx(0.1911676414793058)

    fwfm, dfwfm = pgh.get_fwfm(0.5, hist, bins, var, method="interpolate")
    assert fwfm == approx(1.9897727272727268)
    assert dfwfm == approx(1.0740819016912582)

    fwfm, dfwfm = pgh.get_fwfm(0.5, hist, bins, var, method="fit_slopes")
    assert fwfm == approx(2.2320859865745684)
    assert dfwfm == approx(0.14298871443488284)


def test_get_fwhm():
    from numpy.random import normal

    np.random.seed(42)
    hist, bins, var = pgh.get_hist(normal(size=10000), bins=100, range=(-5, 5))

    fwfm, dfwfm = pgh.get_fwhm(hist, bins, var, method="bins_over_f")
    assert fwfm == approx(2.4)
    assert dfwfm == approx(0.1911676414793058)

    fwfm, dfwfm = pgh.get_fwhm(hist, bins, var, method="interpolate")
    assert fwfm == approx(1.9897727272727268)
    assert dfwfm == approx(1.0740819016912582)

    fwfm, dfwfm = pgh.get_fwhm(hist, bins, var, method="fit_slopes")
    assert fwfm == approx(2.2320859865745684)
    assert dfwfm == approx(0.14298871443488284)


def test_get_gaussian_guess():
    from numpy.random import normal

    np.random.seed(42)
    hist, bins, var = pgh.get_hist(normal(size=10000), bins=100, range=(-5, 5))

    gauss_params = pgh.get_gaussian_guess(hist, bins)
    assert np.allclose(
        np.array(gauss_params, dtype=np.float64),
        np.array(
            [-0.09999999999999964, 1.0191082802547773, 1047.3555083808512],
            dtype=np.float64,
        ),
        rtol=1e-8,
    )


def test_get_bin_estimates():
    from pygama.math.functions.gauss import nb_gauss

    par = np.array([1, 2])
    bins = np.arange(1, 10)
    estimates = pgh.get_bin_estimates(par, nb_gauss, bins, is_integral=False)

    assert np.allclose(
        np.array(estimates, dtype=np.float32),
        np.array(
            [
                9.69233234e-01,
                7.54839602e-01,
                4.57833362e-01,
                2.16265167e-01,
                7.95595087e-02,
                2.27941809e-02,
                5.08606923e-03,
                8.83826307e-04,
            ],
            dtype=np.float32,
        ),
        rtol=1e-8,
    )
