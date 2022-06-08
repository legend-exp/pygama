"""
pygama convenience functions for fitting binned data
"""
import math

import numpy as np
from iminuit import Minuit, cost
from scipy.optimize import brentq, minimize_scalar

import pygama.math.histogram as pgh
from pygama.math.distributions import (
    gauss_norm,
    gauss_tail_pdf,
    gauss_with_tail_pdf,
    unnorm_step_pdf,
)
from pygama.math.functions import gauss_amp


def fit_binned(func, hist, bins, var=None, guess=None,
             cost_func='LL', Extended=True,  simplex=False, bounds=None, fixed = None):
    """Do a binned fit to a histogram.

    Default is Extended Log Likelihood fit, with option for either Least
    Squares or other cost function.

    Parameters
    ----------
    func : the function to fit, if using LL as method needs to be a cdf
    hist, bins, var : histogrammed data
    guess : initial guess parameters
    cost_func : cost function to use
    Extended : run extended or non extended fit
    simplex : whether to include a round of simpson minimisation before main minimisation
    bounds : list of tuples with bounds can be None, e.g. [(0,None), (0,10)]
    fixed : list of parameter indices to fix

    Returns
    -------
    coeff : array
    error : array
    cov_matrix : array
    """
    if guess is None:
        print("auto-guessing not yet implemented, you must supply a guess.")
        return None, None

    if cost_func == 'LL':

        if var is not None:
            t_arr = np.zeros((len(hist),2))
            t_arr[:,0] = hist
            t_arr[:,1] = var
            hist = t_arr

        if Extended ==True:
            cost_func = cost.ExtendedBinnedNLL(hist,bins,  func)

        else:
            cost_func = cost.BinnedNLL( hist,bins, func)

    elif cost_func == 'Least Squares':

        if var is None:
            var = hist # assume Poisson stats if variances are not provided

        if len(bins) == len(hist)+1:
            bin_centres = pgh.get_bin_centers(bins)

        # skip "okay" bins with content 0 +/- 0
        # if bin content is non-zero but var = 0 let the user see the warning
        zeros = (hist == 0)
        zero_errors = (var == 0)
        mask = ~(zeros & zero_errors)
        hist = hist[mask]
        var = np.sqrt(var[mask])
        xvals = bin_centres[mask]
        cost_func = cost.LeastSquares(xvals, hist,var, func)

    m = Minuit(cost_func, *guess)
    if bounds is not None:
        m.limits = bounds
    if fixed is not None:
        for fix in fixed:
            m.fixed[fix] = True
    if simplex == True:
        m.simplex().migrad()
    else:
        m.migrad()
    m.hesse()
    return m.values, m.errors, m.covariance


def goodness_of_fit(hist, bins, var, func, pars, method='var'):
    """Compute chisq and dof of fit

    Parameters
    ----------
    hist, bins, var : array, array, array or None
        histogram data. var can be None if hist is integer counts
    func : function
        the function that was fit to the hist
    pars : array
        the best-fit pars of func. Assumes all pars are free parameters
    method : str
        Sets the choice of "denominator" in the chi2 sum
        'var': user passes in the variances in var (must not have zeros)
        'Pearson': use func (hist must contain integer counts)
        'Neyman': use hist (hist must contain integer counts and no zeros)

    Returns
    -------
    chisq : float
        the summed up value of chisquared
    dof : int
        the number of degrees of freedom
    """
    # arg checks
    if method == 'var':
        if var is None:
            print("goodness_of_fit: var must be non-None to use method 'var'")
            return 0, 0
        if np.any(var==0):
            print("goodness_of_fit: var cannot contain zeros")
            return 0, 0
    if method == 'Neyman' and np.any(hist==0):
        print("goodness_of_fit: hist cannot contain zeros for Neyman method")
        return 0, 0


    # compute expected values
    yy = func(pgh.get_bin_centers(bins), *pars) * pgh.get_bin_widths(bins)

    if method == 'LR':
        log_lr = 2*np.sum(np.where(hist>0 , yy-hist + hist*np.log((hist+1.e-99) / (yy+1.e-99)), yy-hist))
        dof = len(hist) - len(pars)
        return log_lr, dof

    else:
        # compute chi2 numerator and denominator
        numerator = (hist - yy)**2
        if method == 'var':
            denominator = var
        elif method == 'Pearson':
            denominator = yy
        elif method == 'Neyman':
            denominator = hist
        else:
            print(f"goodness_of_fit: unknown method {method}")
            return 0, 0

        # compute chi2 and dof
        chisq = np.sum(numerator/denominator)
        dof = len(hist) - len(pars)
        return chisq, dof


def poisson_gof(pars, func, hist, bins, integral=None, **kwargs):
    """
    The Poisson likelihood does not give a good GOF until the counts are very
    high and all the poisson stats are roughly gaussian and you don't need it
    anyway. But the G.O.F. is calculable for the Poisson likelihood. So we do
    it here.
    """
    mu = pgh.get_bin_estimates(pars, func, hist, bins, integral, **kwargs)
    return 2.*np.sum(mu + hist*(np.log( (hist+1.e-99) / (mu+1.e-99) ) + 1))


def gauss_mode_width_max(hist, bins, var=None, mode_guess=None, n_bins=5,
                         cost_func='Least Squares', inflate_errors=False, gof_method='var'):
    r"""
    Get the max, mode, and width of a peak based on gauss fit near the max
    Returns the parameters of a gaussian fit over n_bins in the vicinity of the
    maximum of the hist (or the max near mode_guess, if provided). This is
    equivalent to a Taylor expansion around the peak maximum because near its
    maximum a Gaussian can be approximated by a 2nd-order polynomial in x:

    .. math::

        A \exp[ -(x-\mu)^2 / 2 \sigma^2 ]
        \simeq A [ 1 - (x-\mu)^2 / 2 \sigma^2 ]
        = A - (1/2!) (A/\sigma^2) (x-\mu)^2

    The advantage of using a gaussian over a polynomial directly is that the
    gaussian parameters are the ones we care about most for a peak, whereas for
    a poly we would have to extract them after the fit, accounting for
    covariances. The gaussian also better approximates most peaks farther down
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
    n_bins : int (optional)
        The number of bins (including the max bin) to be used in the fit. Also
        used for searching for a max near mode_guess
    cost_func : str (optional)
        Passed to fit_binned()
    inflate_errors : bool (optional)
        If true, the parameter uncertainties are inflated by sqrt(chi2red)
        if it is greater than 1
    gof_method : str (optional)
        method flag for goodness_of_fit

    Returns
    -------
    pars : 3-tuple containing the parameters (mode, sigma, maximum) of the
        gaussian fit

        - mode : the estimated x-position of the maximum
        - sigma : the estimated width of the peak. Equivalent to a gaussian
          width (sigma), but based only on the curvature within n_bins of
          the peak.  Note that the Taylor-approxiamted curvature of the
          underlying function in the vicinity of the max is given by max /
          sigma^2
        - maximum : the estimated maximum value of the peak
    cov : 3x3 matrix of floats
        The covariance matrix for the 3 parameters in pars
    """
    bin_centers = pgh.get_bin_centers(bins)
    if mode_guess is not None: i_0 = pgh.find_bin(mode_guess, bins)
    else:
        i_0 = np.argmax(hist)
        mode_guess = bin_centers[i_0]
    amp_guess = hist[i_0]
    i_0 -= int(np.floor(n_bins/2))
    i_n = i_0 + n_bins
    width_guess = (bin_centers[i_n] - bin_centers[i_0])
    vv = None if var is None else var[i_0:i_n]
    guess = (mode_guess, width_guess, amp_guess)
    try:
        pars, errors, cov = fit_binned(gauss_amp, hist[i_0:i_n], bins[i_0:i_n+1], vv,
                         guess=guess, cost_func=cost_func)
    except:
        return None, None
    if pars[1] < 0: pars[1] = -pars[1]
    if inflate_errors:
        chi2, dof = goodness_of_fit(hist, bins, var, gauss_amp, pars)
        if chi2 > dof: cov *= chi2/dof
    return pars, cov


def gauss_mode_max(hist, bins, var=None, mode_guess=None, n_bins=5, poissonLL=False, inflate_errors=False, gof_method='var'):
    """Alias for gauss_mode_width_max that just returns the max and mode

    See Also
    --------
    gauss_mode_width_max

    Returns
    -------
    pars : 2-tuple with the parameters (mode, maximum) of the gaussian fit
        mode : the estimated x-position of the maximum
        maximum : the estimated maximum value of the peak
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
    if pars is None or cov is None: return None, None
    return pars[::2], cov[::2, ::2] # skips "sigma" rows and columns


def taylor_mode_max(hist, bins, var=None, mode_guess=None, n_bins=5, poissonLL=False):
    """Get the max and mode of a peak based on Taylor exp near the max
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
    poissonLL
        DOCME

    Returns
    -------
    pars : 2-tuple with the parameters (mode, max) of the fit
        mode : the estimated x-position of the maximum
        maximum : the estimated maximum value of the peak
    cov : 2x2 matrix of floats
        The covariance matrix for the 2 parameters in pars

    Examples
    --------
    >>> import pygama.analysis.histograms as pgh
    >>> from numpy.random import normal
    >>> import pygama.analysis.peak_fitting as pgf
    >>> hist, bins, var = pgh.get_hist(normal(size=10000), bins=100, range=(-5,5))
    >>> pgf.taylor_mode_max(hist, bins, var, n_bins=5)
    """
    if mode_guess is not None: i_0 = pgh.find_bin(mode_guess, bins)
    else: i_0 = np.argmax(hist)
    i_0 -= int(np.floor(n_bins/2))
    i_n = i_0 + n_bins
    wts = None if var is None else 1/np.sqrt(var[i_0:i_n])

    pars, cov = np.polyfit(pgh.get_bin_centers(bins)[i_0:i_n], hist[i_0:i_n], 2, w=wts, cov='unscaled')
    mode = -pars[1] / 2 / pars[0]
    maximum = pars[2] - pars[0] * mode**2
    # build the jacobian to compute the output covariance matrix
    jac = np.array( [ [pars[1]/2/pars[0]**2,    -1/2/pars[0],       0],
                      [pars[1]**2/4/pars[0]**2, -pars[1]/2/pars[0], 1] ] )
    cov_jact = np.matmul(cov, jac.transpose())
    cov = np.matmul(jac, cov_jact)
    return (mode, maximum), cov


def radford_fwhm(sigma, htail, tau,  cov = None):
    """
    Return the FWHM of the radford_peak function, ignoring background and step
    components. If calculating error also need the normalisation for the step
    function.
    """
    # optimize this to find max value
    def neg_radford_peak_bgfree(E, sigma, htail, tau):
        return -gauss_with_tail_pdf(np.array([E]), 0, sigma, htail, tau)[0]

    if htail<0 or htail>1:
        print("htail outside allowed limits of 0 and 1")
        raise ValueError

    res = minimize_scalar( neg_radford_peak_bgfree,
                           args=(sigma, htail, tau),
                           bounds=(-sigma-htail, sigma+htail) )
    Emax = res.x
    half_max = -neg_radford_peak_bgfree(Emax, sigma, htail, tau)/2.

    # root find this to find the half-max energies
    def radford_peak_bgfree_halfmax(E, sigma, htail, tau, half_max):
        return gauss_with_tail_pdf(np.array([E]), 0, sigma, htail, tau)[0] - half_max

    try:
        lower_hm = brentq( radford_peak_bgfree_halfmax,
                       -(2.5*sigma/2 + htail*tau), Emax,
                       args = (sigma, htail, tau, half_max) )
    except:
        lower_hm = brentq( radford_peak_bgfree_halfmax,
               -(5*sigma + htail*tau), Emax,
               args = (sigma, htail, tau, half_max) )
    try:
        upper_hm = brentq( radford_peak_bgfree_halfmax,
                       Emax, 2.5*sigma/2,
                       args = (sigma, htail, tau, half_max) )
    except:
        upper_hm = brentq( radford_peak_bgfree_halfmax,
                   Emax, 5*sigma,
                   args = (sigma, htail, tau, half_max) )

    if cov is None: return upper_hm - lower_hm

    #calculate uncertainty
    #amp set to 1, mu to 0, hstep+bg set to 0
    pars = [1,0, sigma, htail, tau,0,0]
    step_norm = 1
    gradmax = radford_parameter_gradient(Emax, pars, step_norm)
    gradmax *= 0.5
    grad1 = radford_parameter_gradient(lower_hm, pars,step_norm)
    grad1 -= gradmax
    grad1 /= radford_peakshape_derivative(lower_hm, pars,step_norm)
    grad2 = radford_parameter_gradient(upper_hm, pars,step_norm)
    grad2 -= gradmax
    grad2 /= radford_peakshape_derivative(upper_hm, pars,step_norm)
    grad2 -= grad1

    fwfm_unc = np.sqrt(np.dot(grad2, np.dot(cov, grad2)))

    return upper_hm - lower_hm, fwfm_unc


def radford_peakshape_derivative(E, pars, step_norm):
    """
    Computes the derivative of the Radford peak shape
    """
    n_sig, mu, sigma, htail, tau, n_bkg, hstep = pars

    sigma = abs(sigma)
    gaus = gauss_norm(E, mu, sigma)
    y = (E-mu)/sigma
    ret = -(1-htail)*(y/sigma)*gaus
    ret -= htail/tau*(-gauss_tail_pdf(np.array([E,E-1]), mu, sigma, tau)[0]+gaus)

    return n_sig*ret - n_bkg*hstep*gaus/step_norm #need norm factor for bkg


def radford_parameter_gradient(E, pars, step_norm):
    """
    Computes the gradient of the Radford parameter
    """
    n_sig, mu, sigma, htail, tau, n_bkg, hstep = pars

    gaus = gauss_norm(np.array([E, E-1]), mu, sigma)[0]
    tailL = gauss_tail_pdf(np.array([E, E-1]), mu, sigma, tau)[0]
    if n_bkg ==0:
        step_f = 0
    else:
        step_f = unnorm_step_pdf(np.array([E, E-1]), mu, sigma, hstep)[0] /step_norm

    #some unitless numbers that show up a bunch
    y = (E-mu)/sigma
    sigtauL = sigma/tau

    g_n_sig = 0.5*(htail*tailL + (1-htail)*gaus)
    g_n_bkg = step_f

    g_hs = n_bkg*math.erfc(y/np.sqrt(2))/step_norm

    g_ht = (n_sig/2)*(tailL-gaus)

    #gradient of gaussian part
    g_mu = (1-htail)*y/sigma*gaus
    g_sigma = (1-htail)*(y*y +-1)/sigma*gaus

    #gradient of low tail, use approximation if necessary
    g_mu += htail/tau*(-tailL+gaus)
    g_sigma += htail/tau*(sigtauL*tailL-(sigtauL-y)*gaus)
    g_tau = -htail/tau*( (1.+sigtauL*y+sigtauL*sigtauL)*tailL - sigtauL*sigtauL*gaus) * n_sig

    g_mu = n_sig*g_mu + (2*n_bkg*hstep*gaus)/step_norm
    g_sigma = n_sig*g_sigma + (2*n_bkg*hstep*gaus*y)/(step_norm*np.sqrt(sigma))

    gradient = g_n_sig, g_mu, g_sigma,g_ht, g_tau, g_n_bkg, g_hs
    return np.array(gradient)
