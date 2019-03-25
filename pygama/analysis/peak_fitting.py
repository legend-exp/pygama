"""
- fit_unbinned (unbinned max likelihood fit to data with given likelihood func)
- fit_binned (regular old binned fit (nonlinear least squares))
- neg_log_like (wrapper to give me neg log likelihoods)
- gauss (Define a gaussian distribution and corresponding neg log likelihood)
- radford_peak (David's peak shape)
- xtalball (power-law tail plus gaussian)
- get_gaussian_guess (Given a hist, gives guesses for mu, sigma, and amplitude)
- get_fwhm (find a FWHM from a hist)
"""

import numpy as np
from scipy.optimize import minimize, curve_fit
from scipy.special import erfc
from scipy.stats import crystalball
import pygama.utils as pgu


#unbinned max likelihood fit to data with given likelihood func
def fit_unbinned(likelihood_func,
                 data,
                 start_guess,
                 min_method="L-BFGS-B",
                 bounds=None):
    result = minimize(
        neg_log_like,  # function to minimize
        x0=start_guess,  # start value
        args=(likelihood_func, data),  # additional arguments for function
        method=min_method,  # minimization method, see docs
        bounds=bounds)
    return result.x


#regular old binned fit (nonlinear least squares)
def fit_binned(likelihood_func,
               hist_data,
               bin_centers,
               start_guess,
               var=None,
               bounds=(-np.inf, np.inf)):
    #data should already be histogrammed.
    sigma = None
    if var is not None: 
        # skip "okay" bins with content 0 +/- 0 to avoid div-by-0 error in curve_fit
        # if bin content is non-zero but var = 0 let the user see the warning
        zeros = (hist_data == 0)
        zero_errors = (var == 0)
        mask = ~(zeros & zero_errors)
        sigma = np.sqrt(var)[mask]
        hist_data = hist_data[mask]
        bin_centers = bin_centers[mask]
    coeff, var_matrix = curve_fit(
        likelihood_func, bin_centers, hist_data, p0=start_guess, sigma=sigma, bounds=bounds)
    return coeff


#regular old binned fit (nonlinear least squares)
def fit_hist(func, hist, bins, var=None, guess=None, 
             poissonLL=False, method=None, bounds=None):
    # hist, bins, var : as in return value of pgu.hist()
    # guess : initial parameter guesses. Should be optional -- we can auto-guess
    #         for many common functions. But not yet implemented.
    # poissonLL : use Poisson stats instead of the Gaussian approximation in
    #             each bin. Requires integer stats. You must use parameter
    #             bounds to make sure that func does not go negative over the
    #             x-range of the histogram.
    # method, bounds : options to pass to scipy.optimize.minimize
    if guess is None:
        print("auto-guessing not yet implemented, you must supply a guess.")
        return
    if poissonLL: 
        if var is not None and not np.array_equal(var,hist):
            print("variances are not appropriate for a poisson-LL fit!")
            return
        result = minimize(neg_poisson_log_like, x0=guess, args=(func, hist, bins), method=method, bounds=bounds)
        pars, cov = result.x, result.hess_inv.todense()
    else: 
        if var is None: var = hist # assume Poisson stats if variances are not provided
        # skip "okay" bins with content 0 +/- 0 to avoid div-by-0 error in curve_fit
        # if bin content is non-zero but var = 0 let the user see the warning
        zeros = (hist == 0)
        zero_errors = (var == 0)
        mask = ~(zeros & zero_errors)
        sigma = np.sqrt(var)[mask]
        hist = hist[mask]
        xvals = pgu.get_bin_centers(bins)[mask]
        if bounds is None: bounds=(-np.inf, np.inf)
        pars, cov = curve_fit(func, xvals, hist, p0=guess, sigma=sigma, bounds=bounds)
    return pars, cov


#Wrapper to give me neg log likelihoods
def neg_log_like(params, likelihood_func, data, **kwargs):
    lnl = -np.sum(np.log(likelihood_func(data, *params, **kwargs)))
    return lnl

#Wrapper to give me poisson neg log likelihoods of a histogram
def neg_poisson_log_like(pars, func, hist, bins, **kwargs):
    # ln[ f(x)^n / n! exp(-f(x) ] = const + n ln(f(x)) - f(x)
    # FIXME: bin expected mean mu estimated by f(bin_center)*bin_width. Should
    # add option to integrate function over bin
    mu = func(pgu.get_bin_centers(bins), *pars, **kwargs)*pgu.get_bin_widths(bins) 
    lnl = np.sum(mu - hist*np.log(mu))
    return lnl


#Define a gaussian distribution and corresponding neg log likelihood
def gauss2(x, mu, sigma, A=1):
    return A * (1. / sigma / np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2. * sigma**2))


#David's peak shape
def radford_peak(x, mu, sigma, hstep, htail, tau, bg0, a=1):
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
    return A * crystalball.pdf(x, beta, m, loc=mu, scale=sigma)


#Given a hist, gives guesses for mu, sigma, and amplitude
def get_gaussian_guess(hist, bin_centers):
    max_idx = np.argmax(hist)
    guess_e = bin_centers[max_idx]
    guess_amp = hist[max_idx]

    #find 50% amp bounds on both sides for a FWHM guess
    guess_sigma = get_fwhm(hist, bin_centers) / 2.355  #FWHM to sigma
    guess_area = guess_amp * guess_sigma * np.sqrt(2 * np.pi)

    return (guess_e, guess_sigma, guess_area)


#find a FWHM froma  hist
def get_fwhm(hist, bin_centers):
    idxs_over_50 = hist > 0.5 * np.amax(hist)
    first_energy = bin_centers[np.argmax(idxs_over_50)]
    last_energy = bin_centers[len(idxs_over_50) - np.argmax(idxs_over_50[::-1])]
    return (last_energy - first_energy)
