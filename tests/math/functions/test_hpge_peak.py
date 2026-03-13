import numpy as np
from scipy.stats import exponnorm, norm

from pygama.math.functions.hpge_peak import hpge_get_fwfm, hpge_peak
from pygama.math.functions.sum_dists import SumDists


def test_hpge_peak_pdf():
    x = np.arange(-10, 10)

    sigma = 0.2
    mu = 2
    tau = 0.1
    htail = 0.75
    hstep = 0.5
    x_lo = np.amin(x)
    x_hi = np.max(x)
    n_sig = 10
    n_bkg = 20

    pars = np.array(
        [x_lo, x_hi, n_sig, mu, sigma, htail, tau, n_bkg, hstep],
        dtype=float,
    )

    assert isinstance(hpge_peak, SumDists)

    y_direct = hpge_peak.get_pdf(x, *pars)
    scipy_exgauss = htail * exponnorm.pdf(
        -1 * x, tau / sigma, -1 * mu, sigma
    )  # to be equivalent to the scipy version, x -> -x, mu -> -mu, k -> k/sigma
    scipy_gauss = (1 - htail) * norm.pdf(x, mu, sigma)

    scipy_step = 1 + hstep * (norm.cdf(x, mu, sigma) * 2 - 1)

    maximum = (np.amax(x) - mu) / sigma
    minimum = (np.amin(x) - mu) / sigma
    normalization = sigma * (
        (
            maximum
            + hstep * maximum * (norm.cdf(maximum) * 2 - 1)
            + hstep * np.exp(-1 * maximum**2) / np.sqrt(np.pi)
        )
        - (
            minimum
            + hstep * minimum * (norm.cdf(minimum) * 2 - 1)
            + hstep * np.exp(-1 * minimum**2) / np.sqrt(np.pi)
        )
    )
    scipy_step = scipy_step / normalization

    scipy_y = n_sig * (scipy_exgauss + scipy_gauss) + n_bkg * scipy_step

    assert np.allclose(y_direct, scipy_y, rtol=1e-8)

    y_sig, y_ext = hpge_peak.pdf_ext(x, *pars)
    assert np.allclose(y_ext, scipy_y, rtol=1e-8)
    assert np.allclose(y_sig, n_sig + n_bkg, rtol=1e-8)


def test_hpge_peak_cdf():
    x = np.arange(-10, 10)

    sigma = 0.2
    mu = 2
    tau = 0.1
    htail = 0.75
    hstep = 0.5
    x_lo = np.amin(x)
    x_hi = np.max(x)
    n_sig = 10
    n_bkg = 20

    pars = np.array(
        [x_lo, x_hi, n_sig, mu, sigma, htail, tau, n_bkg, hstep],
        dtype=float,
    )

    assert isinstance(hpge_peak, SumDists)

    y_direct = hpge_peak.get_cdf(x, *pars)

    scipy_exgauss = htail * (
        1 - exponnorm.cdf(-1 * x, tau / sigma, -1 * mu, sigma)
    )  # to be equivalent to the scipy version, x -> -x, mu -> -mu, k -> k/sigma
    scipy_gauss = (1 - htail) * norm.cdf(x, mu, sigma)

    z = (x - mu) / sigma
    scipy_step = sigma * (
        z
        + hstep * z * (2 * norm.cdf(z, 0, 1) - 1)
        + hstep * np.exp(-1 * z**2) / np.sqrt(np.pi)
    )
    maximum = (np.amax(x) - mu) / sigma
    minimum = (np.amin(x) - mu) / sigma
    normalization = sigma * (
        (
            maximum
            + hstep * maximum * (norm.cdf(maximum) * 2 - 1)
            + hstep * np.exp(-1 * maximum**2) / np.sqrt(np.pi)
        )
        - (
            minimum
            + hstep * minimum * (norm.cdf(minimum) * 2 - 1)
            + hstep * np.exp(-1 * minimum**2) / np.sqrt(np.pi)
        )
    )

    scipy_step /= normalization
    scipy_hstep = scipy_step + (1 - scipy_step[-1])

    scipy_y = n_sig * (scipy_exgauss + scipy_gauss) + n_bkg * scipy_hstep

    assert np.allclose(y_direct, scipy_y, rtol=1e-1)

    y_ext = hpge_peak.cdf_ext(x, *pars)
    assert np.allclose(y_ext, scipy_y, rtol=1e-1)


def test_required_args():
    names = hpge_peak.required_args()
    assert names[0] == "x_lo"
    assert names[1] == "x_hi"
    assert names[2] == "n_sig"
    assert names[3] == "mu"
    assert names[4] == "sigma"
    assert names[5] == "htail"
    assert names[6] == "tau"
    assert names[7] == "n_bkg"
    assert names[8] == "hstep"


def test_name():
    assert hpge_peak.name == "hpge_peak"


def test_hpge_get_fwfm_non_hpge_peak_fallback():
    """Test that hpge_get_fwfm uses frac_max (not FWHM) for non-hpge-peak distributions."""
    from pygama.math.functions.gauss import gaussian

    sigma = 2.0
    mu = 5.0
    # gaussian required_args are (mu, sigma)
    pars = np.array([mu, sigma])

    frac_max_half = 0.5
    frac_max_tenth = 0.1

    # Without covariance: result should depend on frac_max
    result_half = hpge_get_fwfm(gaussian, pars, frac_max=frac_max_half)
    result_tenth = hpge_get_fwfm(gaussian, pars, frac_max=frac_max_tenth)
    expected_half = sigma * 2 * np.sqrt(-2 * np.log(frac_max_half))
    expected_tenth = sigma * 2 * np.sqrt(-2 * np.log(frac_max_tenth))

    assert np.isclose(result_half, expected_half)
    assert np.isclose(result_tenth, expected_tenth)
    # Ensure the two results differ (i.e., frac_max is actually used)
    assert not np.isclose(result_half, result_tenth)

    # With covariance: result and error should both depend on frac_max
    n_pars = len(pars)
    cov = np.eye(n_pars) * 0.01
    result_half_cov, err_half_cov = hpge_get_fwfm(
        gaussian, pars, frac_max=frac_max_half, cov=cov
    )
    result_tenth_cov, err_tenth_cov = hpge_get_fwfm(
        gaussian, pars, frac_max=frac_max_tenth, cov=cov
    )
    sigma_idx = np.where(np.array(gaussian.required_args()) == "sigma")[0][0]
    expected_err = np.sqrt(cov[sigma_idx][sigma_idx]) * 2 * np.sqrt(
        -2 * np.log(frac_max_half)
    )

    assert np.isclose(result_half_cov, expected_half)
    assert np.isclose(err_half_cov, expected_err)
    assert not np.isclose(result_half_cov, result_tenth_cov)
