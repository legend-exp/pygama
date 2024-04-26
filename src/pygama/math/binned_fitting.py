"""
pygama convenience functions for fitting binned data
"""

import logging
from typing import Callable, Optional

import numpy as np
from iminuit import Minuit, cost

import pygama.math.histogram as pgh
from pygama.math.functions.gauss import nb_gauss_amp

log = logging.getLogger(__name__)


def fit_binned(
    func: Callable,
    hist: np.ndarray,
    bins: np.ndarray,
    var: np.ndarray = None,
    guess: np.ndarray = None,
    cost_func: str = "LL",
    extended: bool = True,
    simplex: bool = False,
    bounds: tuple[tuple[float, float], ...] = None,
    fixed: tuple[int, ...] = None,
) -> tuple[np.ndarray, ...]:
    """
    Do a binned fit to a histogram.

    Default is extended Log Likelihood fit, with option for either Least
    Squares or other cost function.

    Parameters
    ----------
    func
        the function to fit, if using LL as method needs to be a cdf
    hist, bins, var
        histogrammed data
    guess
        initial guess parameters
    cost_func
        cost function to use
    extended
        run extended or non extended fit
    simplex
        whether to include a round of simpson minimisation before main minimisation
    bounds
        list of tuples with bounds can be None, e.g. [(0,None), (0,10)]
    fixed
        list of parameter indices to fix

    Returns
    -------
    coeff
        Returned fit parameters
    error
        Iminuit errors
    cov_matrix
        Covariance matrix
    """
    if guess is None:
        raise NotImplementedError(
            "auto-guessing not yet implemented, you must supply a guess."
        )

    if cost_func == "LL":

        if var is not None:
            t_arr = np.zeros((len(hist), 2))
            t_arr[:, 0] = hist
            t_arr[:, 1] = var
            hist = t_arr

        if extended is True:
            cost_func = cost.ExtendedBinnedNLL(hist, bins, func)

        else:
            cost_func = cost.BinnedNLL(hist, bins, func)

    elif cost_func == "Least Squares":

        if var is None:
            var = hist  # assume Poisson stats if variances are not provided

        if len(bins) == len(hist) + 1:
            bin_centres = pgh.get_bin_centers(bins)
        else:
            bin_centres = bins

        # skip "okay" bins with content 0 +/- 0
        # if bin content is non-zero but var = 0 let the user see the warning
        zeros = hist == 0
        zero_errors = var == 0
        mask = ~(zeros & zero_errors)
        hist = hist[mask]
        var = np.sqrt(var[mask])
        xvals = bin_centres[mask]
        cost_func = cost.LeastSquares(xvals, hist, var, func)

    m = Minuit(cost_func, *guess)
    if bounds is not None:
        if isinstance(bounds, dict):
            for key, val in bounds.items():
                m.limits[key] = val
    if fixed is not None:
        for fix in fixed:
            m.fixed[fix] = True
    if simplex is True:
        m.simplex().migrad()
    else:
        m.migrad()
    m.hesse()
    return m.values, m.errors, m.covariance


def goodness_of_fit(
    hist: np.ndarray,
    bins: np.ndarray,
    var: np.ndarray,
    func: Callable,
    pars: np.ndarray,
    method: str = "var",
    scale_bins: bool = False,
) -> tuple[float, int]:
    """
    Compute chisq and dof of fit

    Parameters
    ----------
    hist, bins, var
        histogram data. var can be None if hist is integer counts
    func
        the function that was fit to the hist
    pars
        the best-fit pars of func. Assumes all pars are free parameters
    method
        Sets the choice of "denominator" in the chi2 sum
        'var': user passes in the variances in var (must not have zeros)
        'Pearson': use func (hist must contain integer counts)
        'Neyman': use hist (hist must contain integer counts and no zeros)

    Returns
    -------
    chisq
        the summed up value of chisquared
    dof
        the number of degrees of freedom
    """
    # arg checks
    if method == "var":
        if var is None:
            raise RuntimeError("var must be non-None to use method 'var'")
        if np.any(var == 0):
            raise ValueError("var cannot contain zeros")
    if method == "Neyman" and np.any(hist == 0):
        raise ValueError("hist cannot contain zeros for Neyman method")

    # compute expected values
    yy = func(pgh.get_bin_centers(bins), *pars)
    if scale_bins is True:
        yy *= pgh.get_bin_widths(bins)

    if method == "LR":
        log_lr = 2 * np.sum(
            np.where(
                hist > 0,
                yy - hist + hist * np.log((hist + 1.0e-99) / (yy + 1.0e-99)),
                yy - hist,
            )
        )
        dof = len(hist) - len(pars)
        return log_lr, dof

    else:
        # compute chi2 numerator and denominator
        numerator = (hist - yy) ** 2
        if method == "var":
            denominator = var
        elif method == "Pearson":
            denominator = yy
        elif method == "Neyman":
            denominator = hist
        else:
            raise NameError(f"goodness_of_fit: unknown method {method}")

        # compute chi2 and dof
        chisq = np.sum(numerator / denominator)
        dof = len(hist) - len(pars)
        return chisq, dof


def poisson_gof(
    pars: np.ndarray,
    func: Callable,
    hist: np.ndarray,
    bins: np.ndarray,
    is_integral: bool = False,
    **kwargs,
) -> float:
    """
    Calculate the goodness of fit for the Poisson likelihood.

    Parameters
    ----------
    pars
        The parameters of the function, func
    func
        The function that was fit
    hist
        The data that were fit
    bins
        The bins of the histogram that was fit to
    is_integral
        Tells get_bin_estimates if the function is an integral function

    Returns
    -------
    Poisson G.O.F.

    Notes
    -----
    The Poisson likelihood does not give a good GOF until the counts are very
    high and all the poisson stats are roughly gaussian and you don't need it
    anyway. But the G.O.F. is calculable for the Poisson likelihood. So we do
    it here.
    """
    mu = pgh.get_bin_estimates(pars, func, bins, is_integral, **kwargs)
    return 2.0 * np.sum(mu + hist * (np.log((hist + 1.0e-99) / (mu + 1.0e-99)) + 1))


def gauss_mode_width_max(
    hist: np.ndarray,
    bins: np.ndarray,
    var: Optional[np.ndarray] = None,
    mode_guess: Optional[float] = None,
    n_bins: Optional[int] = 5,
    cost_func: str = "Least Squares",
    inflate_errors: Optional[bool] = False,
    gof_method: Optional[str] = "var",
) -> tuple[np.ndarray, ...]:
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
    hist
        The values of the histogram to be fit
    bins
        The bin edges of the histogram to be fit
    var
        The variances of the histogram values. If not provided, square-root
        variances are assumed.
    mode_guess
        An x-value (not a bin index!) near which a peak is expected. The
        algorithm fits around the maximum within +/- n_bins of the guess. If not
        provided, the center of the max bin of the histogram is used.
    n_bins
        The number of bins (including the max bin) to be used in the fit. Also
        used for searching for a max near mode_guess
    cost_func
        Passed to fit_binned()
    inflate_errors
        If true, the parameter uncertainties are inflated by sqrt(chi2red)
        if it is greater than 1
    gof_method
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
    if mode_guess is not None:
        i_0 = pgh.find_bin(mode_guess, bins)
    else:
        i_0 = np.argmax(hist)
        mode_guess = bin_centers[i_0]
    amp_guess = hist[i_0]
    i_0 -= int(np.floor(n_bins / 2))
    i_n = i_0 + n_bins
    width_guess = bin_centers[i_n] - bin_centers[i_0]
    vv = None if var is None else var[i_0:i_n]
    guess = (mode_guess, width_guess, amp_guess)
    pars, errors, cov = fit_binned(
        nb_gauss_amp,
        hist[i_0:i_n],
        bins[i_0 : i_n + 1],
        vv,
        guess=guess,
        cost_func=cost_func,
    )
    if pars[1] < 0:
        pars[1] = -pars[1]
    if inflate_errors:
        chi2, dof = goodness_of_fit(hist, bins, var, nb_gauss_amp, pars)
        if chi2 > dof:
            cov *= chi2 / dof
    return np.asarray([pars["mu"], pars["sigma"], pars["a"]]), np.asarray(cov)


def gauss_mode_max(
    hist: np.ndarray, bins: np.ndarray, **kwargs
) -> tuple[np.ndarray, ...]:
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
    >>> import pygama.math.histograms as pgh
    >>> from numpy.random import normal
    >>> import pygama.math.peak_fitting as pgf
    >>> hist, bins, var = pgh.get_hist(normal(size=10000), bins=100, range=(-5,5))
    >>> pgf.gauss_mode_max(hist, bins, var=var, n_bins=20)
    """
    pars, cov = gauss_mode_width_max(hist, bins, **kwargs)
    if pars is None or cov is None:
        raise RuntimeError("fit binned failed to work")
    return pars[::2], cov[::2, ::2]  # skips "sigma" rows and columns


def gauss_mode(hist: np.ndarray, bins: np.ndarray, **kwargs) -> tuple[float, float]:
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
    if pars is None or cov is None:
        return None, None
    return pars[0], np.sqrt(cov[0, 0])


def taylor_mode_max(
    hist: np.ndarray,
    bins: np.ndarray,
    var: Optional[np.ndarray] = None,
    mode_guess: Optional[float] = None,
    n_bins: int = 5,
) -> tuple[np.ndarray, ...]:
    """Get the max and mode of a peak based on Taylor exp near the max
    Returns the amplitude and position of a peak based on a poly fit over n_bins
    in the vicinity of the maximum of the hist (or the max near mode_guess, if provided)

    Parameters
    ----------
    hist
        The values of the histogram to be fit. Often: send in a slice around a peak
    bins
        The bin edges of the histogram to be fit
    var
        The variances of the histogram values. If not provided, square-root
        variances are assumed.
    mode_guess
        An x-value (not a bin index!) near which a peak is expected. The
        algorithm fits around the maximum within +/- n_bins of the guess. If not
        provided, the center of the max bin of the histogram is used.
    n_bins
        The number of bins (including the max bin) to be used in the fit. Also
        used for searching for a max near mode_guess

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
    if mode_guess is not None:
        i_0 = pgh.find_bin(mode_guess, bins)
    else:
        i_0 = np.argmax(hist)
    i_0 -= int(np.floor(n_bins / 2))
    i_n = i_0 + n_bins
    wts = None if var is None else 1 / np.sqrt(var[i_0:i_n])

    pars, cov = np.polyfit(
        pgh.get_bin_centers(bins)[i_0:i_n], hist[i_0:i_n], 2, w=wts, cov="unscaled"
    )
    mode = -pars[1] / 2 / pars[0]
    maximum = pars[2] - pars[0] * mode**2
    # build the jacobian to compute the output covariance matrix
    jac = np.array(
        [
            [pars[1] / 2 / pars[0] ** 2, -1 / 2 / pars[0], 0],
            [pars[1] ** 2 / 4 / pars[0] ** 2, -pars[1] / 2 / pars[0], 1],
        ]
    )
    cov_jact = np.matmul(cov, jac.transpose())
    cov = np.matmul(jac, cov_jact)
    return (mode, maximum), cov
