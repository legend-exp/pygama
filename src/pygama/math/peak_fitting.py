import math
import sys

import numba as nb
import numpy as np
from iminuit import Minuit, cost
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import crystalball

import pygama.math.histogram as pgh

limit = np.log(sys.float_info.max)/10
kwd = {"parallel": False, "fastmath": True}

def fit_hist(func, hist, bins, var=None, guess=None,
             poissonLL=False, integral=None, method=None, bounds=None):
    """
    .. deprecated:: 0.8
        Replaced by :func:`.fit_binned`. Will be removed in future releases.

    do a binned fit to a histogram (nonlinear least squares).
    can either do a poisson log-likelihood fit (jason's favourite) or
    use curve_fit w/ an arbitrary function.

    Parameters
    ----------
    func
        function to be fitted
    hist, bins, var
        as in return value of pygama.histograms.get_hist()
    guess
        initial parameter guesses. Should be optional -- we can auto-guess for
        many common functions. But not yet implemented.
    poissonLL
        use Poisson stats instead of the Gaussian approximation in each bin.
        Requires integer stats. You must use parameter bounds to make sure that
        func does not go negative over the x-range of the histogram.
    integral
        DOCME
    method, bounds
        options to pass to :func:`scipy.optimize.minimize`

    Returns
    -------
    coeff, cov_matrix : tuple(array, matrix)
    """

    print("Fit hist is now deprecated use fit_binned instead")
    print("Rerouting to fit_binned, bounds will not be applied")

    coeff, errors, cov_matrix = fit_binned(func, hist, bins, var=var, guess=guess, cost_func='Least Squares', bounds = None)

    return coeff, cov_matrix

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
        else:
             bin_centres = bins
        # skip "okay" bins with content 0 +/- 0
        # if bin content is non-zero but var = 0 let the user see the warning
        zeros = (hist == 0)
        zero_errors = (var == 0)
        mask = ~(zeros & zero_errors)
        hist = hist[mask]
        var = np.sqrt(var[mask])
        xvals = bin_centres[mask]
        cost_func = cost.LeastSquares(xvals, hist, var, func)

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

def fit_unbinned(func, data, guess=None,
             Extended=True, cost_func = 'LL',simplex=False,
             bounds=None, fixed=None):
    """Do a unbinned fit to data.
    Default is Extended Log Likelihood fit, with option for other cost functions.

    Parameters
    ----------
    func : the function to fit
    data : the data
    guess : initial guess parameters
    Extended : run extended or non extended fit
    cost_func : cost function to use
    simplex : whether to include a round of simpson minimisation before main minimisation
    bounds : list of tuples with bounds can be None, e.g. [(0,None), (0,10)]
    fixed : list of parameter indices to fix
    coeff, cov_matrix : tuple(array, matrix)
    """
    if guess is None:
        print("auto-guessing not yet implemented, you must supply a guess.")
        return None, None

    if cost_func =='LL':
        if Extended ==True:
            cost_func = cost.ExtendedUnbinnedNLL(data, func)

        else:
            cost_func = cost.UnbinnedNLL(data, func)

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

def goodness_of_fit(hist, bins, var, func, pars, method='var', scale_bins=False):
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
    yy = func(pgh.get_bin_centers(bins), *pars) 
    if scale_bins ==True:
        yy*= pgh.get_bin_widths(bins)

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

def get_bin_estimates(pars, func, hist, bins, integral=None, **kwargs):
    """
    Bin expected means are estimated by f(bin_center)*bin_width. Supply an
    integrating function to compute the integral over the bin instead.
    TODO: make default integrating function a numerical method that is off by
    default.
    """
    if integral is None:
        return func(pgh.get_bin_centers(bins), *pars, **kwargs) * pgh.get_bin_widths(bins)
    else:
        return integral(bins[1:], *pars, **kwargs) - integral(bins[:-1], *pars, **kwargs)


def poisson_gof(pars, func, hist, bins, integral=None, **kwargs):
    """
    The Poisson likelihood does not give a good GOF until the counts are very
    high and all the poisson stats are roughly gaussian and you don't need it
    anyway. But the G.O.F. is calculable for the Poisson likelihood. So we do
    it here.
    """
    mu = get_bin_estimates(pars, func, hist, bins, integral, **kwargs)
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
    pars : ndarray containing the parameters (mode, sigma, maximum) of the gaussian fit
        - mode : the estimated x-position of the maximum
        - sigma : the estimated width of the peak. Equivalent to a gaussian
          width (sigma), but based only on the curvature within n_bins of
          the peak.  Note that the Taylor-approxiamted curvature of the
          underlying function in the vicinity of the max is given by max /
          sigma^2
        - maximum : the estimated maximum value of the peak
    cov : 3x3 ndarray of floats
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
    pars, errors, cov = fit_binned(gauss_amp, hist[i_0:i_n], bins[i_0:i_n+1], vv, guess=guess, cost_func=cost_func)
    if pars[1] < 0: pars[1] = -pars[1]
    if inflate_errors:
        chi2, dof = goodness_of_fit(hist, bins, var, gauss_amp, pars)
        if chi2 > dof: cov *= chi2/dof
    return np.asarray([pars['mu'], pars['sigma'], pars['a']]), np.asarray(cov)


def gauss_mode_max(hist, bins, **kwargs):
    """Alias for gauss_mode_width_max that just returns the max and mode

    See Also
    --------
    gauss_mode_width_max

    Returns
    -------
    pars : ndarray with the parameters (mode, maximum) of the gaussian fit
        mode : the estimated x-position of the maximum
        maximum : the estimated maximum value of the peak
    cov : 2x2 ndarray of floats
        The covariance matrix for the 2 parameters in pars

    Examples
    --------
    >>> import pygama.math.histogram as pgh
    >>> from numpy.random import normal
    >>> import pygama.math.peak_fitting as pgf
    >>> hist, bins, var = pgh.get_hist(normal(size=10000), bins=100, range=(-5,5))
    >>> pgf.gauss_mode_max(hist, bins, var=var, n_bins=20)
    """
    pars, cov = gauss_mode_width_max(hist, bins, **kwargs)
    if pars is None or cov is None: return None, None
    return pars[::2], cov[::2, ::2] # skips "sigma" rows and columns


def gauss_mode(hist, bins, **kwargs):
    """Alias for gauss_mode_max that just returns the mode (position) of a peak

    See Also
    --------
    gauss_mode_max
    gauss_mode_width_max

    Returns
    -------
    mode : the estimated x-position of the maximum
    dmode : the uncertainty in the mode
    """
    pars, cov = gauss_mode_width_max(hist, bins, **kwargs)
    if pars is None or cov is None: return None, None
    return pars[0], np.sqrt(cov[0,0])


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

@nb.njit(**kwd)
def nb_erf(x):
    """
    Numba version of error function
    """

    y = np.empty_like(x)
    for i in nb.prange(len(x)):
        y[i] = math.erf(x[i])
    return y

@nb.njit(**kwd)
def nb_erfc(x):
    """
    Numba version of complementary error function
    """

    y = np.empty_like(x)
    for i in nb.prange(len(x)):
        y[i] = math.erfc(x[i])
    return y

@nb.njit(**kwd)
def gauss(x, mu, sigma):
    """
    Gaussian, unnormalised for use in building pdfs, w/ args: mu, sigma.
    """

    if sigma ==0: invs=np.nan
    else: invs = 1.0 / sigma
    z = (x - mu) * invs
    return np.exp(-0.5 * z ** 2)

@nb.njit(**kwd)
def gauss_norm(x, mu, sigma):
    """
    Normalised Gaussian, w/ args: mu, sigma.
    """

    if sigma ==0: invs=np.nan
    else: invs = 1.0 / sigma
    z = (x - mu) * invs
    invnorm = invs/ np.sqrt(2 * np.pi)
    return np.exp(-0.5 * z ** 2) * invnorm

@nb.njit(**kwd)
def gauss_cdf(x, mu, sigma):
    """
    gaussian cdf, w/ args: mu, sigma.
    """
    return 1/2 * (1 + nb_erf((x - mu)/(sigma*np.sqrt(2))))

@nb.njit(**kwd)
def gauss_amp(x, mu, sigma, a):
    """
    Gaussian with height as a parameter for fwhm etc. args mu sigma, amplitude
    """
    return a * gauss(x,mu,sigma)

@nb.njit(**kwd)
def gauss_pdf(x, mu, sigma, n_sig):
    """
    Basic Gaussian pdf args; mu, sigma, n_sig (number of signal events)
    """

    return n_sig * gauss_norm(x,mu,sigma)


def gauss_uniform(x, n_sig, mu, sigma, n_bkg, components = False):
    """
    define a gaussian signal on a uniform background,
    args: n_sig mu, sigma for the signal and n_bkg for the background
    """

    if components==False:
        return 1/(np.nanmax(x)-np.nanmin(x)) * n_bkg + n_sig * gauss_norm(x,mu,sigma)
    else:
        return n_sig * gauss_norm(x,mu,sigma), 1/(np.nanmax(x)-np.nanmin(x)) * n_bkg


def gauss_linear(x, n_sig, mu, sigma, n_bkg, b, m, components=False):
    """
    gaussian signal + linear background function
    args: n_sig mu, sigma for the signal and n_bkg,b,m for the background
    """


    norm = (m/2 *np.nanmax(x)**2 + b*np.nanmax(x)) - (m/2 *np.nanmin(x)**2 + b*np.nanmin(x))

    if components==False:
        return n_bkg/norm * (m * x + b) + n_sig * gauss_norm(x, mu, sigma)
    else:
        return  n_sig * gauss_norm(x, mu, sigma), n_bkg/norm * (m * x + b)

@nb.njit(**kwd)
def step_int(x,mu,sigma, hstep):
    """
    Integral of step function w/args mu, sigma, hstep
    """

    part1 = x+hstep*(x-mu)*nb_erf((x-mu)/(np.sqrt(2)*sigma))
    part2 = - np.sqrt(2/np.pi)*hstep*sigma*gauss(x,mu,sigma)
    return  part1-part2

@nb.njit(**kwd)
def unnorm_step_pdf(x,  mu, sigma, hstep):
    """
    Unnormalised step function for use in pdfs
    """

    invs = (np.sqrt(2)*sigma)
    z = (x-mu)/invs
    step_f = 1 + hstep * nb_erf(z)
    return step_f

@nb.njit(**kwd)
def step_pdf(x,  mu, sigma, hstep, lower_range=np.inf , upper_range=np.inf):
    """
    Normalised step function w/args mu, sigma, hstep
    Can be used as a component of other fit functions
    """

    step_f = unnorm_step_pdf(x,  mu, sigma, hstep)
    if lower_range ==np.inf and upper_range ==np.inf:
        integral = step_int(np.array([np.nanmin(x), np.nanmax(x)]), mu, sigma, hstep)
    else:
        integral = step_int(np.array([lower_range, upper_range]), mu, sigma, hstep)

    norm = integral[1]-integral[0]
    return step_f/norm

@nb.njit(**kwd)
def step_cdf(x,mu,sigma, hstep, lower_range=np.inf , upper_range=np.inf):
    """
    CDF for step function w/args mu, sigma, hstep
    """

    cdf = step_int(x,mu,sigma,hstep)
    if lower_range ==np.inf and upper_range ==np.inf:
        integral = step_int(np.array([np.nanmin(x), np.nanmax(x)]), mu, sigma, hstep)
    else:
        integral = step_int(np.array([lower_range, upper_range]), mu, sigma, hstep)
    norm = integral[1]-integral[0]
    cdf =  (1/norm) * cdf
    c = 1-cdf[-1]
    return cdf+c

def gauss_step_pdf(x,  n_sig, mu, sigma, n_bkg, hstep, lower_range=np.inf , upper_range=np.inf, components=False):
    """
    Pdf for Gaussian on step background
    args: n_sig mu, sigma for the signal and n_bkg,hstep for the background
    """

    try:
        bkg= step_pdf(x, mu, sigma, hstep, lower_range, upper_range)
    except ZeroDivisionError:
        bkg = np.zeros_like(x, dtype=np.float64)
    if np.any(bkg<0):
            bkg= np.zeros_like(x, dtype=np.float64)
    pdf = n_sig*gauss_norm(x,mu,sigma) +\
          n_bkg*bkg
    if components ==False:
        return pdf
    else:
        return n_sig*gauss_norm(x,mu,sigma), n_bkg*bkg

def extended_gauss_step_pdf(x,  n_sig, mu, sigma, n_bkg, hstep, lower_range=np.inf , upper_range=np.inf, components=False):
    """
    Pdf for Gaussian on step background for Compton spectrum, returns also the total number of events for extended unbinned fits
    args: n_sig mu, sigma for the signal and n_bkg, hstep for the background
    """

    if components ==False:
        return n_sig+n_bkg , gauss_step_pdf(x,  n_sig, mu, sigma, n_bkg, hstep, lower_range, upper_range)
    else:
        sig, bkg = gauss_step_pdf(x,  n_sig, mu, sigma, n_bkg, hstep,lower_range, upper_range, components=True)
        return n_sig+n_bkg, sig, bkg

def gauss_step_cdf(x,  n_sig, mu, sigma,n_bkg, hstep, lower_range=np.inf , upper_range=np.inf, components=False):
    """
    Cdf for Gaussian on step background
    args: n_sig mu, sigma for the signal and n_bkg,hstep for the background
    """
    try:
        bkg = step_cdf(x, mu, sigma, hstep, lower_range, upper_range)
    except ZeroDivisionError:
        bkg = np.zeros_like(x, dtype=np.float64)
    if np.any(bkg<0):
        bkg= np.zeros_like(x, dtype=np.float64)
    if components ==False:
        cdf = (1/(n_sig+n_bkg))*(n_sig*gauss_cdf(x, mu, sigma) +\
          n_bkg*bkg)
        return cdf
    else:
        return (1/(n_sig+n_bkg))*n_sig*gauss_cdf(x, mu, sigma), (1/(n_sig+n_bkg))*(n_bkg*bkg)

@nb.njit(**kwd)
def gauss_tail_pdf(x, mu, sigma, tau):
    """
    A gaussian tail function template
    Can be used as a component of other fit functions w/args mu,sigma,tau
    """

    x = np.asarray(x)
    tmp = ((x-mu)/tau) + ((sigma**2)/(2*tau**2))
    tail_f = np.where(tmp < limit,
                      gauss_tail_exact(x, mu, sigma, tau),
                      gauss_tail_approx(x, mu, sigma, tau))
    return tail_f

@nb.njit(**kwd)
def gauss_tail_exact(x, mu, sigma, tau):
    tmp = ((x-mu)/tau) + ((sigma**2)/(2*tau**2))
    abstau = np.absolute(tau)
    tmp = np.where(tmp < limit, tmp, limit)
    z = (x-mu)/sigma
    tail_f = (1/(2*abstau)) * np.exp(tmp) * nb_erfc( (tau*z + sigma)/(np.sqrt(2)*abstau))
    return tail_f

@nb.njit(**kwd)
def gauss_tail_approx(x, mu, sigma, tau):
    den = 1/(sigma + tau*(x-mu)/sigma)
    tail_f = sigma * gauss_norm(x, mu, sigma) * den * (1.-tau*tau*den*den)
    return tail_f

@nb.njit(**kwd)
def gauss_tail_integral(x,mu,sigma,tau):
    """
    Integral for gaussian tail
    """

    abstau = np.abs(tau)
    part1 = (tau/(2*abstau)) * nb_erf((tau*(x-mu) )/(np.sqrt(2)*sigma*abstau))
    part2 =    tau * gauss_tail_pdf(x,mu,sigma,tau)
    return part1+part2

@nb.njit(**kwd)
def gauss_tail_norm(x,mu,sigma,tau, lower_range=np.inf , upper_range=np.inf):
    """
    Normalised gauss tail. Note: this is only needed when the fitting range
    does not include the whole tail
    """

    tail = gauss_tail_pdf(x,mu,sigma,tau)
    if lower_range ==np.inf and upper_range ==np.inf:
        integral = gauss_tail_integral(np.array([np.nanmin(x), np.nanmax(x)]), mu, sigma, tau)
    else:
        integral = gauss_tail_integral(np.array([lower_range, upper_range]), mu, sigma, tau)
    norm = integral[1]-integral[0]
    return tail/norm

@nb.njit(**kwd)
def gauss_tail_cdf(x,mu,sigma,tau, lower_range=np.inf , upper_range=np.inf):
    """
    CDF for gaussian tail
    """

    cdf = gauss_tail_integral(x,mu,sigma,tau)
    if lower_range ==np.inf and upper_range ==np.inf:
        integral = gauss_tail_integral(np.array([np.nanmin(x), np.nanmax(x)]), mu, sigma, tau)
    else:
        integral = gauss_tail_integral(np.array([lower_range, upper_range]), mu, sigma, tau)
    norm = integral[1]-integral[0]
    cdf =  (1/norm) * cdf
    c = 1-cdf[-1]
    return cdf+c

def gauss_with_tail_pdf(x, mu, sigma,  htail,tau, components=False):
    """
    Pdf for gaussian with tail
    """

    if htail < 0 or htail > 1:
        if components ==False:
            return np.full_like(x, np.nan, dtype='float64')
        else:
            return np.full_like(x, np.nan, dtype='float64'), np.full_like(x, np.nan, dtype='float64')

    peak = gauss_norm(x,mu,sigma)
    try:
        tail = gauss_tail_pdf(x, mu, sigma, tau)
    except ZeroDivisionError:
        tail = np.zeros_like(x, dtype=np.float64)
    if components ==False:
        return (1-htail)*peak + htail*tail
    else:
        return (1-htail)*peak, htail*tail

def gauss_with_tail_cdf(x, mu, sigma, htail,  tau, components=False):
    """
    Cdf for gaussian with tail
    """

    if htail < 0 or htail > 1:
        if components ==False:
            return np.full_like(x, np.nan, dtype='float64')
        else:
            return np.full_like(x, np.nan, dtype='float64'), np.full_like(x, np.nan, dtype='float64')

    peak = gauss_cdf(x,mu,sigma)
    try:
        tail = gauss_tail_cdf(x, mu, sigma, tau)
    except  ZeroDivisionError:
        tail = np.zeros_like(x, dtype=np.float64)
    if components==False:
        return (1-htail)*peak + htail*tail
    else:
        return (1-htail)*peak, htail*tail

def radford_pdf(x, n_sig, mu, sigma, htail, tau, n_bkg, hstep,
                lower_range=np.inf , upper_range=np.inf,  components=False):
    """
    David Radford's HPGe peak shape PDF consists of a gaussian with tail signal
    on a step background
    """

    try:
        bkg= step_pdf(x, mu, sigma, hstep, lower_range, upper_range)
    except ZeroDivisionError:
        bkg = np.zeros_like(x, dtype=np.float64)
    if np.any(bkg<0):
        bkg = np.zeros_like(x, dtype=np.float64)
    if components ==False:
        sig = gauss_with_tail_pdf(x, mu, sigma, htail,  tau)
        pdf = (n_bkg * bkg +\
             n_sig *  sig)
        return pdf
    else:
        peak, tail = gauss_with_tail_pdf(x, mu, sigma, htail,  tau, components=components)
        return n_sig *peak, n_sig*tail, n_bkg * bkg

def extended_radford_pdf(x, n_sig, mu, sigma, htail, tau, n_bkg, hstep,
                         lower_range=np.inf , upper_range=np.inf, components=False):
    """
    Pdf for gaussian with tail signal and step background, also returns number of events
    """

    if components ==False:
        return n_sig + n_bkg, radford_pdf(x, n_sig,  mu, sigma, htail, tau, n_bkg, hstep, lower_range, upper_range)
    else:
        peak, tail, bkg = radford_pdf(x, n_sig,  mu, sigma, htail, tau, n_bkg, hstep,
                                      lower_range, upper_range,components=components)
        return n_sig + n_bkg, peak, tail, bkg

def radford_cdf(x, n_sig, mu, sigma, htail, tau, n_bkg, hstep, lower_range=np.inf , upper_range=np.inf,  components=False):
    """
    Cdf for gaussian with tail signal and step background
    """
    try:
        bkg = step_cdf(x, mu, sigma, hstep, lower_range, upper_range)
    except ZeroDivisionError:
        bkg = np.zeros_like(x, dtype=np.float64)
    if np.any(bkg<0):
        bkg= np.zeros_like(x, dtype=np.float64)
    if components ==False:
        sig = gauss_with_tail_cdf(x, mu, sigma, htail)
        pdf = (1/(n_sig+n_bkg))*(n_sig*gauss_with_tail_cdf(x, mu, sigma, htail,tau) +\
            n_bkg*bkg)
        return pdf
    else:
        peak, tail = gauss_with_tail_cdf(x, mu, sigma, htail, components= True)
        return (n_sig/(n_sig+n_bkg))*peak, (n_sig/(n_sig+n_bkg))*tail, (n_bkg/(n_sig+n_bkg))*bkg

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
        #print("htail outside allowed limits of 0 and 1")
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
    n_sig, mu, sigma, htail, tau, n_bkg, hstep = pars

    sigma = abs(sigma)
    gaus = gauss_norm(E, mu, sigma)
    y = (E-mu)/sigma
    ret = -(1-htail)*(y/sigma)*gaus
    ret -= htail/tau*(-gauss_tail_pdf(np.array([E,E-1]), mu, sigma, tau)[0]+gaus)

    return n_sig*ret - n_bkg*hstep*gaus/step_norm #need norm factor for bkg

def radford_parameter_gradient(E, pars, step_norm):
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

def get_mu_func(func, pars, cov = None, errors=None):

    if  func == gauss_step_cdf or func == gauss_step_pdf or func == extended_gauss_step_pdf:
        if len(pars) ==5:
            n_sig, mu, sigma, n_bkg, hstep = pars
        elif len(pars) ==7:
            n_sig, mu, sigma, n_bkg, hstep, low_range, high_range = pars
        if errors is not None:
            return mu, errors[1]
        elif cov is not None:
            return mu, np.sqrt(cov[1][1])
        else:
            return mu

    elif  func == radford_cdf or func == radford_pdf or func == extended_radford_pdf:
        if len(pars) ==7:
            n_sig, mu, sigma, htail, tau, n_bkg, hstep = pars
        elif len(pars) ==9:
            n_sig, mu, sigma, htail, tau, n_bkg, hstep, low_range, high_range = pars
        if errors is not None:
            return mu, errors[1]
        elif cov is not None:
            return mu, np.sqrt(cov[1][1])
        else:
            return mu

    else:
        print(f'get_mu_func not implemented for {func.__name__}')
        return None

def get_fwhm_func(func, pars, cov = None):

    if  func == gauss_step_cdf or func == gauss_step_pdf or func == extended_gauss_step_pdf:
        if len(pars) ==5:
            n_sig, mu, sigma, n_bkg, hstep = pars
        elif len(pars) ==7:
            n_sig, mu, sigma, n_bkg, hstep, low_range, high_range = pars
        if cov is None:
            return sigma*2*np.sqrt(2*np.log(2))
        else:
            return sigma*2*np.sqrt(2*np.log(2)), np.sqrt(cov[2][2])*2*np.sqrt(2*np.log(2))

    elif  func == radford_cdf or func == radford_pdf or func == extended_radford_pdf:
        if len(pars) ==7:
            n_sig, mu, sigma, htail, tau, n_bkg, hstep = pars
        elif len(pars) ==9:
            n_sig, mu, sigma, htail, tau, n_bkg, hstep, low_range, high_range = pars

        return radford_fwhm(sigma, htail, tau, cov)
    else:
        print(f'get_fwhm_func not implemented for {func.__name__}')
        return None

def get_total_events_func(func, pars, cov = None, errors=None):

    if  func == gauss_step_cdf or func == gauss_step_pdf or func == extended_gauss_step_pdf:
        if len(pars) ==5:
            n_sig, mu, sigma, n_bkg, hstep = pars
        elif len(pars) ==7:
            n_sig, mu, sigma, n_bkg, hstep, low_range, high_range = pars
        if errors is not None:
            return n_sig+n_bkg, np.sqrt(errors[0]**2 + errors[3]**2)
        elif cov is not None:
            return n_sig+n_bkg, np.sqrt(cov[0][0]**2 + cov[3][3]**2)
        else:
            return n_sig+n_bkg

    elif  func == radford_cdf or func == radford_pdf or func == extended_radford_pdf:
        if len(pars) ==7:
            n_sig, mu, sigma, htail, tau, n_bkg, hstep = pars
        elif len(pars) ==9:
            n_sig, mu, sigma, htail, tau, n_bkg, hstep, low_range, high_range = pars
        if errors is not None:
            return n_sig+n_bkg, np.sqrt(errors[0]**2 + errors[5]**2)
        elif cov is not None:
            return n_sig+n_bkg, np.sqrt(cov[0][0]**2 + cov[5][5]**2)
        else:
            return n_sig+n_bkg
    else:
        print(f'get_total_events_func not implemented for {func.__name__}')
        return None

def Am_double(x,  n_sig1, mu1, sigma1,  n_sig2, mu2,sigma2, n_sig3, mu3,sigma3, n_bkg1, hstep1, n_bkg2, hstep2,
             lower_range=np.inf , upper_range=np.inf, components=False):
    """
    A Fit function exclusevly for a 241Am 99keV and 103keV lines situation
    Consists of

     - three gaussian peaks (two lines + one bkg line in between)
     - two steps (for the two lines)
     - two tails (for the two lines)
    """
    bkg1 = n_bkg1*step_pdf(x, mu1, sigma1, hstep1, lower_range, upper_range )
    bkg2 = n_bkg2*step_pdf(x, mu2, sigma2, hstep2, lower_range, upper_range)
    if np.any(bkg1<0) or np.any(bkg2<0):
        return 0, np.zeros_like(x)
    sig1 = n_sig1*gauss_norm(x,mu1,sigma1)
    sig2 = n_sig2* gauss_norm(x,mu2,sigma2)
    sig3 = n_sig3* gauss_norm(x,mu3,sigma3)
    if components ==False:
        return sig1+sig2+sig3+bkg1+bkg2
    else:
        return sig1,sig2,sig3,bkg1,bkg2

def extended_Am_double(x,  n_sig1, mu1, sigma1,  n_sig2, mu2,sigma2, n_sig3, mu3,sigma3,
                       n_bkg1, hstep1, n_bkg2, hstep2,
                     lower_range=np.inf , upper_range=np.inf, components=False):
    if components ==False:
        return n_sig1+n_sig2+n_sig3 + n_bkg1+n_bkg2, Am_double(n_sig1, mu1, sigma1,  n_sig2, mu2,sigma2,
                                                               n_sig3, mu3,sigma3,
                                                               n_bkg1, hstep1, n_bkg2, hstep2,
                                                                 lower_range, upper_range)
    else:
        sig1,sig2,sig3,bkg1,bkg2 = Am_double(n_sig1, mu1, sigma1,  n_sig2, mu2,sigma2, n_sig3, mu3,sigma3,
                                             n_bkg1, hstep1, n_bkg2, hstep2,
                                             lower_range , upper_range,components=components)
        return n_sig1+n_sig2+n_sig3 + n_bkg1+n_bkg2, sig1,sig2,sig3,bkg1,bkg2


def double_gauss_pdf(x,  n_sig1,  mu1, sigma1, n_sig2, mu2,sigma2,n_bkg,hstep,
                     lower_range=np.inf, upper_range=np.inf, components=False):
    """
    A Fit function exclusevly for a 133Ba 81keV peak situation
    Consists of

     - two gaussian peaks (two lines)
     - one step
    """
    bkg = n_bkg*step_pdf(x, mu1, sigma1, hstep, lower_range, upper_range)
    if np.any(bkg<0):
        return 0, np.zeros_like(x)
    sig1 = n_sig1*gauss_norm(x,mu1,sigma1)
    sig2 = n_sig2* gauss_norm(x,mu2,sigma2)
    if components == False:
        return sig1 + sig2 + bkg
    else:
        return sig1, sig2, bkg

def extended_double_gauss_pdf(x,  n_sig1,  mu1, sigma1, n_sig2, mu2,sigma2,n_bkg,hstep,
                     lower_range=np.inf , upper_range=np.inf, components=False):
    """
    A Fit function exclusevly for a 133Ba 81keV peak situation
    Consists of

     - two gaussian peaks (two lines)
     - one step
    """

    if components == False:
        pdf = double_gauss_pdf(x,  n_sig1,  mu1, sigma1, n_sig2, mu2,sigma2,n_bkg,hstep,
                     lower_range, upper_range)
        return n_sig1+n_sig2+n_bkg, pdf
    else:
        sig1, sig2, bkg = double_gauss_pdf(x,  n_sig1,  mu1, sigma1, n_sig2, mu2,sigma2,n_bkg,hstep,
                     lower_range, upper_range,components=components)
        return n_sig1+n_sig2+n_bkg, sig1, sig2, bkg

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


def poly(x, pars):
    """
    A polynomial function with pars following the polyfit convention
    """
    result = x*0 # do x*0 to keep shape of x (scalar or array)
    if len(pars) == 0: return result
    result += pars[-1]
    for i in range(1, len(pars)):
        result += pars[-i-1]*x
        x = x*x
    return result
