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

    Returns
    ------
    coeff, cov_matrix : tuple(array, matrix)
    """
    if guess is None:
        print("auto-guessing not yet implemented, you must supply a guess.")
        return None, None

    if poissonLL:
        if var is not None and not np.array_equal(var, hist):
            print("variances are not appropriate for a poisson-LL fit!")
            return None, None

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


def goodness_of_fit(hist, bins, func, p_fit):
    """
    compute reduced chisq and fwhm_err for 
    """
    chisq = []
    for i, h in enumerate(hist):
        model = func(bins[i], *p_fit)
        diff = (model - h)**2 / model
        chisq.append(abs(diff))
    rchisq = sum(np.array(chisq) / len(hist))
    return rchisq


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
    high and all the poisson stats are roughly gaussian and you don't need it
    anyway. But the G.O.F. is calculable for the Poisson likelihood. So we do
    it here.
    """
    mu = get_bin_estimates(pars, func, hist, bins, integral, **kwargs)
    return 2.*np.sum(mu + hist*(np.log( (hist+1.e-99) / (mu+1.e-99) ) + 1))


def gauss_mode_width_max(hist, bins, var=None, mode_guess=None, n_bins=5, poissonLL=False):
    """ Get the max, mode, and width of a peak based on gauss fit near the max

    Returns the parameters of a gaussian fit over n_bins in the vicinity of the
    maximum of the hist (or the max near mode_guess, if provided). This is
    equivalent to a Taylor expansion around the peak maximum because near its
    maximum a Gaussian can be approximated by a 2nd-order polynomial in x:

    A exp[ -(x-mu)^2 / 2 sigma^2 ] ~= A [ 1 - (x-mu)^2 / 2 sigma^2 ]
                                    = A - (1/2!) (A/sigma^2) (x-mu)^2

    The advantage of using a gaussian over a polynomial directly is that the
    gaussian parameters are the ones we care about most for a peak, whereas for
    a poly we would have to extract them after the fit, accounting for
    covariances. The guassian also better approximates most peaks farther down
    the peak. However, the gauss fit is nonlinear and thus less stable.

    Parameters
    ----------
    hist : array-like
        The values of the histogram to be fit
    bins : array-like
        The bin edges of the histogram to be fit
    var : array-like (optional)
        The variances of the histogram values. If not provided, square-root
        variances are assumed.
    mode_guess : float (optional)
        An x-value (not a bin index!) near which a peak is expected. The
        algorithm fits around the maximum within +/- n_bins of the guess. If not
        provided, the center of the max bin of the histogram is used.
    n_bins : int
        The number of bins (including the max bin) to be used in the fit. Also
        used for searching for a max near mode_guess

    Returns
    -------
    (pars, cov) : tuple (array, matrix)
        pars : 3-tuple containing the parameters (mode, sigma, maximum) of the
               gaussian fit
            mode : the estimated x-position of the maximum
            sigma : the estimated width of the peak. Equivalent to a guassian
                width (sigma), but based only on the curvature within n_bins of
                the peak.  Note that the Taylor-approxiamted curvature of the
                underlying function in the vicinity of the max is given by max /
                sigma^2
            maximum : the estimated maximum value of the peak
        cov : 3x3 matrix of floats
            The covariance matrix for the 3 parameters in pars
    """

    bin_centers = ph.get_bin_centers(bins)
    if mode_guess is not None: i_0 = ph.find_bin(mode_guess, bins)
    else:
        i_0 = np.argmax(hist) 
        mode_guess = bin_centers[i_0]
    amp_guess = hist[i_0]
    i_0 -= int(np.floor(n_bins/2))
    i_n = i_0 + n_bins
    width_guess = (bin_centers[i_n] - bin_centers[i_0])
    vv = None if var is None else var[i_0:i_n]
    guess = (mode_guess, width_guess, amp_guess)
    pars, cov = fit_hist(gauss_basic, hist[i_0:i_n], bins[i_0:i_n+1], vv,
                         guess=guess, poissonLL=poissonLL)
    if pars[1] < 0: pars[1] = -pars[1]
    return pars, cov


def gauss_mode_max(hist, bins, var=None, mode_guess=None, n_bins=5, poissonLL=False):
    """ Alias for gauss_mode_width_max that just returns the max and mode 

    Parameters
    --------
    See gauss_mode_width_max

    Returns
    -------
    (pars, cov) : tuple (array, matrix)
        pars : 2-tuple with the parameters (maximum, mode) of the gaussian fit     
            maximum : the estimated maximum value of the peak
            mode : the estimated x-position of the maximum
        cov : 2x2 matrix of floats
            The covariance matrix for the 2 parameters in pars

    Examples
    --------
    >>> import pygama.analysis.histograms as pgh
    >>> from numpy.random import normal
    >>> import pygama.analysis.peak_fitting as pgf
    >>> hist, bins, var = pgh.get_hist(normal(size=10000), bins=100, range=(-5,5))
    >>> pgf.gauss_mode_max(hist, bins, var, n_bins=20)
    """
    pars, cov = gauss_mode_width_max(hist, bins, var, mode_guess, n_bins, poissonLL)
    return pars[::2], cov[::2, ::2] # skips "sigma" rows and columns



def taylor_mode_max(hist, bins, var=None, mode_guess=None, n_bins=5, poissonLL=False):
    """ Get the max and mode of a peak based on Taylor exp near the max

    Returns the amplitude and position of a peak based on a poly fit over n_bins
    in the vicinity of the maximum of the hist (or the max near mode_guess, if provided)

    Parameters
    ----------
    hist : array-like
        The values of the histogram to be fit. Often: send in a slice around a peak
    bins : array-like
        The bin edges of the histogram to be fit
    var : array-like (optional)
        The variances of the histogram values. If not provided, square-root
        variances are assumed.
    mode_guess : float (optional)
        An x-value (not a bin index!) near which a peak is expected. The
        algorithm fits around the maximum within +/- n_bins of the guess. If not
        provided, the center of the max bin of the histogram is used.
    n_bins : int
        The number of bins (including the max bin) to be used in the fit. Also
        used for searching for a max near mode_guess

    Returns
    -------
    (maximum, mode) : tuple (float, float)
        maximum : the estimated maximum value of the peak
        mode : the estimated x-position of the maximum

    Examples
    --------
    >>> import pygama.analysis.histograms as pgh
    >>> from numpy.random import normal
    >>> import pygama.analysis.peak_fitting as pgf
    >>> hist, bins, var = pgh.get_hist(normal(size=10000), bins=100, range=(-5,5))
    >>> pgf.taylor_mode_max(hist, bins, var, n_bins=5)
    """

    if mode_guess is not None: i_0 = ph.find_bin(mode_guess, bins)
    else: i_0 = np.argmax(hist) 
    i_0 -= int(np.floor(n_bins/2))
    i_n = i_0 + n_bins
    wts = None if var is None else 1/np.sqrt(var[i_0:i_n])

    pars, cov = np.polyfit(ph.get_bin_centers(bins)[i_0:i_n], hist[i_0:i_n], 2, w=wts, cov='unscaled')
    mode = -pars[1] / 2 / pars[0]
    maximum = pars[2] - pars[0] * mode**2
    # build the jacobian to compute the output covariance matrix
    jac = np.array( [ [pars[1]/2/pars[0]**2,    -1/2/pars[0],       0],
                      [pars[1]**2/4/pars[0]**2, -pars[1]/2/pars[0], 1] ] )
    cov_jact = np.matmul(cov, jac.transpose())
    cov = np.matmul(jac, cov_jact)
    return (mode, maximum), cov


def gauss_basic(x, mu, sigma, height=1, C=0):
    """
    define a gaussian distribution, w/ args: mu, sigma, height
    (behaves differently than gauss() in fits)
    """
    return height * np.exp(-(x - mu)**2 / (2. * sigma**2)) + C


def gauss(x, mu, sigma, A=1, const=0):
    """
    define a gaussian distribution, w/ args: mu, sigma, area, const.
    """
    height = A / sigma / np.sqrt(2 * np.pi)
    return gauss_basic(x, mu, sigma, height, const)


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


def gauss_bkg(x, a, mu, sigma, bkg): # deprecate this?
    """
    gaussian + const background function
    """
    return gauss(x, mu, sigma, a, bkg)


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


def gauss_tail(x,mu, sigma, tail,tau):
    """
    A gaussian tail function template
    Can be used as a component of other fit functions
    """
    tail_f = tail/(2*tau) * np.exp( (x-mu)/tau + sigma**2/(np.sqrt(2) * tau)**2) * erfc( (x-mu)/(np.sqrt(2)*sigma) + sigma/(np.sqrt(2)*tau))
    return tail_f


def step(x, mu, sigma, bkg, a):
    """
    A step function template
    Can be used as a component of other fit functions
    """
    step_f = bkg + a * erfc((x-mu)/(np.sqrt(2)*sigma))
    return step_f


def gauss_step(x, a, mu, sigma, bkg, s, components=False):
    """
    gaussian + step function for Compton spectrum
    """
    peak_f = gauss(x,mu,sigma,a)
    step_f = step(x,mu,sigma,bkg,s)

    peak = peak_f + step_f

    if components:
      return peak_f, step_f
    else:
      return peak


def gauss_cdf(x, a, mu, sigma, tail, tau, bkg, s, components=False):
    """
    I guess this should be similar to radford_peak (peak + tail + step)
    This is how I used it in root peak fitting scripts
    """ 
    peak_f = gauss(x, mu, sigma, a)
    tail_f = gauss_tail(x, mu, sigma, tail, tau)
    step_f = step(x, mu, sigma, bkg, s)

    peak = peak_f + tail_f + step_f

    if components:
      return peak, tail_f, step_f, peak_f
    else:
      return peak


def Am_double(x,a1,mu1,sigma1,a2,mu2,sigma2,a3,mu3,sigma3,b1,b2,s1,s2,
              components=False) :
    """
    A Fit function exclusevly for a 241Am 99keV and 103keV lines situation 
    Consists of 
     - three gaussian peaks (two lines + one bkg line in between)
     - two steps (for the two lines)
     - two tails (for the two lines)
    """

    step1 = step(x,mu1,sigma1,b1,s1)
    step2 = step(x,mu2,sigma2,b2,s2)
  
    gaus1 = gauss(x,mu1,sigma1,a1)
    gaus2 = gauss(x,mu2,sigma2,a2)
    gaus3 = gauss(x,mu3,sigma3,a3)

    #tail1 = gauss_tail(x,mu1,sigma1,t1,tau1)
    #tail2 = gauss_tail(x,mu2,sigma2,t2,tau2)
    double_f = step1 + step2 + gaus1 + gaus2 + gaus3# + tail1 + tail2  

    if components:
       return double_f, gaus1, gaus2, gaus3, step1, step2#, tail1, tail2 
    else:
       return double_f


def double_gauss(x,a1,mu1,sigma1,a2,mu2,sigma2,b1,s1,components=False) :
    """
    A Fit function exclusevly for a 133Ba 81keV peak situation 
    Consists of 
     - two gaussian peaks (two lines)
     - one step
     """

    step1 = step(x,mu1,sigma1,b1,s1)
    #step2 = step(x,mu2,sigma2,b2,s2)

    gaus1 = gauss(x,mu1,sigma1,a1)
    gaus2 = gauss(x,mu2,sigma2,a2)
    #gaus3 = gauss(x,mu3,sigma3,a3)

    #tail1 = gauss_tail(x,mu1,sigma1,t1,tau1)
    #tail2 = gauss_tail(x,mu2,sigma2,t2,tau2)
    double_f = step1 +  gaus1 + gaus2  

    if components:
       return double_f, gaus1, gaus2, step1  
    else:
       return double_f


def xtalball(x, mu, sigma, A, beta, m):
    """
    power-law tail plus gaussian https://en.wikipedia.org/wiki/Crystal_Ball_function
    """
    return A * crystalball.pdf(x, beta, m, loc=mu, scale=sigma)


def cal_slope(x, m1, m2):
    """
    Fit the calibration values
    """
    return np.sqrt(m1 +(m2/(x**2)))

