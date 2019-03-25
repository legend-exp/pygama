import numpy as np
from scipy.optimize import minimize, curve_fit
from scipy.special import erfc
from scipy.stats import crystalball
import pygama.analysis.histograms as pgh


def fit_hist(func, hist, bins, var=None, guess=None,
             poissonLL=False, method=None, bounds=None):
    """
    do a binned fit to a histogram (nonlinear least squares).
    can either do a poisson log-likelihood fit (jason's fave) or
    use curve_fit w/ an arbitrary function.

    - hist, bins, var : as in return value of pygama.utils.get_hist()
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
                          args=(func, hist, bins),
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
        xvals = pgh.get_bin_centers(bins)[mask]
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
    # jason says this is deprecated.
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


def neg_poisson_log_like(pars, func, hist, bins, **kwargs):
    """
    Wrapper to give me poisson neg log likelihoods of a histogram
        ln[ f(x)^n / n! exp(-f(x) ] = const + n ln(f(x)) - f(x)
    Note: bin expected mean mu estimated by f(bin_center)*bin_width. Should
    TODO: add option to integrate function over bin
    """
    mu = func(pgh.get_bin_centers(bins), *pars, **kwargs) * pgh.get_bin_widths(bins)
    return np.sum(mu - hist*np.log(mu))
<<<<<<< HEAD

#Define a gaussian distribution and corresponding neg log likelihood
def gauss(x, mu, sigma, A=1):
    """
    define a gaussian distribution, w/ args: mu, sigma, area (optional).
    TODO: return the corresponding neg log likelihood
    """
    return A * (1. / sigma / np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2. * sigma**2))


def radford_peak(x, mu, sigma, hstep, htail, tau, bg0, a=1):
    """
    David Radford's HPGe peak shape function
    """
    #make sure the fractional amplitude parameters stay reasonable...
    if htail < 0 or htail > 1: return np.zeros_like(x)
    if hstep < 0 or hstep > 1: return np.zeros_like(x)

    bg_term = bg0  #+ x*bg1
    if np.any(bg_term < 0): return np.zeros_like(x)
    step = a * hstep * erfc((x - mu) / (sigma * np.sqrt(2)))
    le_tail = a * htail * erfc(
        (x - mu) / (sigma * np.sqrt(2)) + sigma / (tau * np.sqrt(2))) * np.exp(
            (x - mu) / tau) / (2 * tau * np.exp(-(sigma /
                                                  (np.sqrt(2) * tau))**2))
    return (1 - htail) * gauss(x, mu, sigma, a) + bg_term + step + le_tail


# power-law tail plus gaussian https://en.wikipedia.org/wiki/Crystal_Ball_function
def xtalball(x, mu, sigma, A, beta, m):
    """
    power-law tail plus gaussian https://en.wikipedia.org/wiki/Crystal_Ball_function
    """
    return A * crystalball.pdf(x, beta, m, loc=mu, scale=sigma)


