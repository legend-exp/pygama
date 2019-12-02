import numpy as np
from scipy.optimize import minimize, curve_fit
from scipy.special import erf, erfc, gammaln
from scipy.stats import crystalball

import pygama.analysis.histograms as ph


def fit_hist(func, hist, bins, var=None, guess=None,
             poissonLL=False, integral=None, method=None, bounds=None):
    """
    do a binned fit to a histogram (nonlinear least squares).
    can either do a poisson log-likelihood fit (jason's fave) or
    use curve_fit w/ an arbitrary function.

    - hist, bins, var : as in return value of pygama.histograms.get_hist()
    - guess : initial parameter guesses. Should be optional -- we can auto-guess
              for many common functions. But not yet implemented.
    - poissonLL : use Poisson stats instead of the Gaussian approximation in
                  each bin. Requires integer stats. You must use parameter
                  bounds to make sure that func does not go negative over the
                  x-range of the histogram.
    - method, bounds : options to pass to scipy.optimize.minimize
    """
    if guess is None:
        print("auto-guessing not yet implemented, you must supply a guess.")
        return

    if poissonLL:
        if var is not None and not np.array_equal(var, hist):
            print("variances are not appropriate for a poisson-LL fit!")
            return

        if method is None:
            method = "L-BFGS-B"

        result = minimize(neg_poisson_log_like, x0=guess,
                          args=(func, hist, bins, integral),
                          method=method, bounds=bounds)

        coeff, cov_matrix = result.x, result.hess_inv.todense()

    else:
        if var is None:
            var = hist # assume Poisson stats if variances are not provided

        # skip "okay" bins with content 0 +/- 0 to avoid div-by-0 error in curve_fit
        # if bin content is non-zero but var = 0 let the user see the warning
        zeros = (hist == 0)
        zero_errors = (var == 0)
        mask = ~(zeros & zero_errors)
        sigma = np.sqrt(var)[mask]
        hist = hist[mask]
        xvals = ph.get_bin_centers(bins)[mask]
        if bounds is None:
            bounds = (-np.inf, np.inf)

        coeff, cov_matrix = curve_fit(func, xvals, hist,
                                      p0=guess, sigma=sigma, bounds=bounds)

    return coeff, cov_matrix


def neg_log_like(params, f_likelihood, data, **kwargs):
    """
    given a likelihood function and data, return the negative log likelihood.
    """
    return -np.sum(np.log(f_likelihood(data, *params, **kwargs)))


def fit_unbinned(f_likelihood, data, start_guess, min_method=None, bounds=None):
    """
    unbinned max likelihood fit to data with given likelihood func
    """
    if method is None:
        method="L-BFGS-B" # minimization method, see docs

    result = minimize(
        neg_log_like, # function to minimize
        x0 = start_guess, # start value
        args = (f_likelihood, data),
        method = min_method,
        bounds = bounds)

    return result.x


def fit_binned(f_likelihood, hist, bin_centers, start_guess, var=None, bounds=None):
    """
    regular old binned fit (nonlinear least squares). data should already be
    histogrammed (see e.g. pygama.analysis.histograms.get_hist)
    # jason says this is deprecated. Use ph.fit_hist() instead.
    """
    sigma = None
    if bounds is None:
        bounds = (-np.inf, np.inf)

    # skip "okay" bins with content 0 +/- 0 to avoid div-by-0 error in curve_fit
    # if bin content is non-zero but var = 0 let the user see the warning
    if var is not None:
        zeros = (hist == 0)
        zero_errors = (var == 0)
        mask = ~(zeros & zero_errors)
        sigma = np.sqrt(var)[mask]
        hist = hist[mask]
        bin_centers = bin_centers[mask]

    # run curve_fit
    coeff, var_matrix = curve_fit(f_likelihood, bin_centers, hist,
                                  p0=start_guess, sigma=sigma, bounds=bounds)
    return coeff


def get_bin_estimates(pars, func, hist, bins, integral=None, **kwargs):
    """
    Bin expected means are estimated by f(bin_center)*bin_width. Supply an
    integrating function to compute the integral over the bin instead.
    TODO: make default integrating function a numerical method that is off by
    default.
    """
    if integral is None:
        return func(ph.get_bin_centers(bins), *pars, **kwargs) * ph.get_bin_widths(bins)
    else:
        return integral(bins[1:], *pars, **kwargs) - integral(bins[:-1], *pars, **kwargs)


def neg_poisson_log_like(pars, func, hist, bins, integral=None, **kwargs):
    """
    Wrapper to give me poisson neg log likelihoods of a histogram
        ln[ f(x)^n / n! exp(-f(x) ] = const + n ln(f(x)) - f(x)
    """
    mu = get_bin_estimates(pars, func, hist, bins, integral, **kwargs)
    
    # func and/or integral should never give a negative value: let negative
    # values cause errors that get passed to the user. However, mu=0 is okay,
    # but causes problems for np.log(). When mu is zero there had better not be
    # any counts in the bins. So use this to pull the fit like crazy.
    return np.sum(mu - hist*np.log(mu+1.e-99))


def poisson_gof(pars, func, hist, bins, integral=None, **kwargs):
    """
    The Poisson likelihood does not give a good GOF until the counts are very
    high and all the poisson stats are roughly guassian and you don't need it
    anyway. But the G.O.F. is calculable for the Poisson likelihood. So we do
    it here.
    """
    mu = get_bin_estimates(pars, func, hist, bins, integral, **kwargs)
    return 2.*np.sum(mu + hist*(np.log( (hist+1.e-99) / (mu+1.e-99) ) + 1))


def gauss(x, mu, sigma, A=1, C=0):
    """
    define a gaussian distribution, w/ args: mu, sigma, area, const.
    """
    norm = A / sigma / np.sqrt(2 * np.pi)
    return norm * np.exp(-(x - mu)**2 / (2. * sigma**2)) + C


def gauss_int(x, mu, sigma, A=1):
    """
    integral of a gaussian from 0 to x, w/ args: mu, sigma, area, const.
    """
    return A/2 * (1 + erf((x - mu)/sigma/np.sqrt(2)))


def gauss_lin(x, mu, sigma, a, b, m):
    """
    gaussian + linear background function
    """
    return m * x + b + gauss(x, mu, sigma, a)


def radford_peak(x, mu, sigma, hstep, htail, tau, bg0, a=1, components=False):
    """
    David Radford's HPGe peak shape function
    """
    # make sure the fractional amplitude parameters stay reasonable
    if htail < 0 or htail > 1: 
        return np.zeros_like(x)
    if hstep < 0 or hstep > 1: 
        return np.zeros_like(x)

    bg_term = bg0  #+ x*bg1
    if np.any(bg_term < 0): 
        return np.zeros_like(x)

    # compute the step and the low energy tail
    step = a * hstep * erfc((x - mu) / (sigma * np.sqrt(2)))
    le_tail = a * htail
    le_tail *= erfc((x - mu) / (sigma * np.sqrt(2)) + sigma / (tau * np.sqrt(2)))
    le_tail *= np.exp((x - mu) / tau)
    le_tail /= (2 * tau * np.exp(-(sigma / (np.sqrt(2) * tau))**2))

    if not components:
        # add up all the peak shape components
        return (1 - htail) * gauss(x, mu, sigma, a) + bg_term + step + le_tail
    else:
        # return individually to make a pretty plot
        return (1 - htail), gauss(x, mu, sigma, a), bg_term, step, le_tail


def xtalball(x, mu, sigma, A, beta, m):
    """
    power-law tail plus gaussian https://en.wikipedia.org/wiki/Crystal_Ball_function
    """
    return A * crystalball.pdf(x, beta, m, loc=mu, scale=sigma)


