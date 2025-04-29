"""
pygama convenience functions for 1D histograms.

1D hists in pygama require 3 things available from all implementations
of 1D histograms of numerical data in python: hist, bins, and var:
- hist: an array of histogram values
- bins: an array of bin edges
- var: an array of variances in each bin
If var is not provided, pygama assuems that the hist contains "counts" with
variance = counts (Poisson stats)

These are just convenience functions, provided for your convenience. Hopefully
they will help you if you need to do something trickier than is provided (e.g.
2D hists).
"""

import logging
from typing import Callable, Optional, Union

import hist as bh
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
from matplotlib import rcParams

import pygama.math.utils as pgu

log = logging.getLogger(__name__)


def get_hist(
    data: np.ndarray,
    bins: Optional[Union[int, np.ndarray, str]] = None,
    range: Optional[tuple[float, float]] = None,
    dx: Optional[float] = None,
    wts: Optional[Union[float, np.ndarray]] = None,
) -> tuple[np.ndarray, ...]:
    """return hist, bins, var after binning data

    This is just a wrapper for humba.histogram, with optional weights for each
    element and proper computing of variances.

    Note: there are no overflow / underflow bins.

    Available binning methods:

      - Default (no binning arguments) : 100 bins over an auto-detected range
      - bins=n, range=(x_lo, x_hi) : n bins over the specified range (or leave
        range=None for auto-detected range)
      - bins=[str] : use one of np.histogram's automatic binning algorithms
      - bins=bin_edges_array : array lower bin edges, supports non-uniform binning
      - dx=dx, range=(x_lo, x_hi): bins of width dx over the specified range.
        Note: dx overrides the bins argument!

    Parameters
    ----------
    data
        The array of data to be histogrammed
    bins
        int: the number of bins to be used in the histogram
        array: an array of bin edges to use
        str: the name of the np.histogram automatic binning algorithm to use
        If not provided, humba.histogram's default auto-binning routine is used
    range
        (x_lo, x_high) is the tuple of low and high x values to uses for the
        very ends of the bin range. If not provided, np.histogram chooses the
        ends based on the data in data
    dx
        Specifies the bin width. Overrides "bins" if both arguments are present
    wts
        Array of weights for each bin. For example, if you want to divide all
        bins by a time T to get the bin contents in count rate, set wts = 1/T.
        Variances will be computed for each bin that appropriately account for
        each data point's weighting.

    Returns
    -------
    hist
        the values in each bin of the histogram
    bins
        an array of bin edges: bins[i] is the lower edge of the ith bin.
        Note: it includes the upper edge of the last bin and does not
        include underflow or overflow bins. So len(bins) = len(hist) + 1
    var
        array of variances in each bin of the histogram
    """
    if bins is None:
        bins = 100  # override boost_histogram.Histogram default of just 10

    if dx is not None:
        bins = int((range[1] - range[0]) / dx)

    if range is None:
        range = [np.amin(data), np.amax(data)]

    # bins includes left edge of first bin and right edge of all other bins
    # allow scalar weights
    if wts is not None and np.shape(wts) == ():
        wts = np.full_like(data, wts)

    # initialize the boost_histogram object
    if isinstance(bins, int):
        boost_histogram = bh.Hist(
            bh.axis.Regular(bins=bins, start=range[0], stop=range[1]),
            storage=bh.storage.Weight(),
        )
    else:
        # if bins are specified need to use variable
        boost_histogram = bh.Hist(bh.axis.Variable(bins), storage=bh.storage.Weight())
    # create the histogram
    boost_histogram.fill(data, weight=wts)
    # read out the histogram, bins, and variances
    hist, bins = boost_histogram.to_numpy()
    var = boost_histogram.variances()

    return hist, bins, var


def better_int_binning(
    x_lo: float = 0, x_hi: float = None, dx: float = None, n_bins: float = None
) -> tuple[int, int, int, int]:
    """Get a good binning for integer data.

    Guarantees an integer bin width.

    At least two of x_hi, dx, or n_bins must be provided.

    Parameters
    ----------
    x_lo
        Desired low x value for the binning
    x_hi
        Desired high x value for the binning
    dx
        Desired bin width
    n_bins
        Desired number of bins

    Returns
    -------
    x_lo
        int values for best x_lo
    x_hi
        int values for best x_hi, returned if x_hi is not None
    dx
        best int bin width, returned if arg dx is not None
    n_bins
        best int n_bins, returned if arg n_bins is not None
    """
    # process inputs
    n_nones = int(x_hi is None) + int(dx is None) + int(n_bins is None)
    if n_nones > 1:
        raise ValueError("must provide two of x_hi, dx or n_bins")
    if n_nones == 0:
        log.warning("Overconstrained. Ignoring x_hi.")
        x_hi = None

    # get valid dx or n_bins
    if dx is not None:
        if dx <= 0:
            raise ValueError(f"invalid dx={dx}")
        dx = np.round(dx)
        if dx == 0:
            dx = 1
    if n_bins is not None:
        if n_bins <= 0:
            raise ValueError(f"invalid n_bins={n_bins}")
        n_bins = np.round(n_bins)

    # can already return if no x_hi
    if x_hi is None:  # must have both dx and n_bins
        return int(x_lo), int(dx), int(n_bins)

    # x_hi is valid. Get a valid dx if we don't have one
    if dx is None:  # must have n_bins
        dx = np.round((x_hi - x_lo) / n_bins)
    if dx == 0:
        dx = 1

    # Finally, build a good binning from dx
    final_n_bins = np.ceil((x_hi - x_lo) / dx)
    x_lo = np.floor(x_lo)
    x_hi = x_lo + final_n_bins * dx
    if n_bins is None:
        return int(x_lo), int(x_hi), int(dx)
    else:
        return int(x_lo), int(x_hi), int(final_n_bins)


@nb.njit(parallel=False, fastmath=True)
def get_bin_centers(bins: np.ndarray) -> np.ndarray:
    """
    Returns an array of bin centers from an input array of bin edges.
    Works for non-uniform binning. Note: a new array is allocated

    Parameters
    ----------
    bins
        The input array of bin-edges

    Returns
    ----------
    bin_centers
        The array of bin centers
    """
    return (bins[:-1] + bins[1:]) / 2.0


@nb.njit(parallel=False, fastmath=True)
def get_bin_widths(bins: np.ndarray) -> np.ndarray:
    """
    Returns an array of bin widths from an input array of bin edges.
    Works for non-uniform binning.

    Parameters
    ----------
    bins
        The input array of bin-edges

    Returns
    ----------
    bin_widths
        An array of bin widths
    """
    return bins[1:] - bins[:-1]


@nb.njit(parallel=False, fastmath=True)
def find_bin(x: float, bins: np.ndarray) -> int:
    """
    Returns the index of the bin containing x
    Returns -1 for underflow, and len(bins) for overflow
    For uniform bins, jumps directly to the appropriate index.
    For non-uniform bins, binary search is used.

    Parameters
    ----------
    x
        The value to search for amongst the bins
    bins
        The input array of bin-edges

    Returns
    -------
    bin_idx
        Index of the bin containing x

    TODO: replace np.searchsorted with for loops, numba will speed this function up then
    """
    # first handle overflow / underflow
    if len(bins) == 0:
        return 0  # i.e. overflow
    if x < bins[0]:
        return -1
    if x > bins[-1]:
        return len(bins)

    # we are definitely in range, and there are at least 2 bin edges, one below
    # and one above x. try assuming uniform bins
    dx = bins[1] - bins[0]
    index = int(np.floor((x - bins[0]) / dx))
    if bins[index] <= x and bins[index + 1] > x:
        return index

    # bins are non-uniform: find by binary search
    return np.searchsorted(bins, x, side="right")


def range_slice(
    x_min: float,
    x_max: float,
    hist: np.ndarray,
    bins: np.ndarray,
    var: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, ...]:
    """
    Get the histogram bins and values for a given slice of the range

    Parameters
    ----------
    x_min
        Lower endpoint of the range
    x_min
        Upper endpoint of the range
    hist, bins, var
        The histogrammed data to search through

    See Also
    --------
    find_bin
        for parameters and return values
    """
    i_min = find_bin(x_min, bins)
    i_max = find_bin(x_max, bins)
    if var is not None:
        var = var[i_min:i_max]
    return hist[i_min:i_max], bins[i_min : i_max + 1], var


def get_fwhm(
    hist: np.ndarray,
    bins: np.ndarray,
    var: Optional[np.ndarray] = None,
    mx: Optional[Union[float, tuple[float, float]]] = None,
    dmx: Optional[float] = 0,
    bl: Optional[Union[float, tuple[float, float]]] = 0,
    dbl: Optional[float] = 0,
    method: str = "bins_over_f",
    n_slope: int = 3,
) -> tuple[float, float]:
    """Convenience function for the FWHM of data in a histogram

    Typically used by sending slices around a peak. Searches about argmax(hist)
    for the peak to fall by [fraction] from mx to bl

    Parameters
    ----------
    fraction
        The fractional amplitude at which to evaluate the full width
    hist
        The histogram data array containing the peak
    bins
        An array of bin edges for the histogram
    var
        An array of histogram variances. Used with the 'fit_slopes' method
    mx
        The value to use for the max of the peak. If None, np.amax(hist) is
        used.
    dmx
        The uncertainty in mx
    bl
        Used to specify an offset from which to estimate the FWFM.
    dbl
        The uncertainty in the bl
    method
        'bins_over_f' : the simplest method: just take the difference in the bin
            centers that are over [fraction] of max. Only works for high stats and
            FWFM/bin_width >> 1
        'interpolate' : interpolate between the bins that cross the [fraction]
            line.  Works well for high stats and a reasonable number of bins.
            Uncertainty incorporates var, if provided.
        'fit_slopes' : fit over n_slope bins in the vicinity of the FWFM and
            interpolate to get the fractional crossing point. Works okay even
            when stats are moderate but requires enough bins that dx traversed
            by n_slope bins is approximately linear. Incorporates bin variances
            in fit and final uncertainties if provided.
    n_slope
        Number of bins in the vicinity of the FWFM used to interpolate the fractional
        crossing point with the 'fit_slopes' method

    Returns
    -------
    fwhm, dfwhm
        fwfm: the full width at half of the maximum above bl
        dfwfm: the uncertainty in fwhm


    See Also
    --------
    get_fwfm
        Function that computes the FWFM
    """
    if len(bins) == len(hist):
        log.warning(
            "note: this function has been updated to require bins rather than bin_centers. Don't trust this result"
        )
    return get_fwfm(0.5, hist, bins, var, mx, dmx, bl, dbl, method, n_slope)


def get_fwfm(
    fraction: float,
    hist: np.ndarray,
    bins: np.ndarray,
    var: Optional[np.ndarray] = None,
    mx: Optional[Union[float, tuple[float, float]]] = None,
    dmx: Optional[float] = 0,
    bl: Optional[Union[float, tuple[float, float]]] = 0,
    dbl: Optional[float] = 0,
    method: str = "bins_over_f",
    n_slope: int = 3,
) -> tuple[float, float]:
    """
    Estimate the full width at some fraction of the max of data in a histogram

    Typically used by sending slices around a peak. Searches about argmax(hist)
    for the peak to fall by [fraction] from mx to bl

    Parameters
    ----------
    fraction
        The fractional amplitude at which to evaluate the full width
    hist
        The histogram data array containing the peak
    bins
        An array of bin edges for the histogram
    var
        An array of histogram variances. Used with the 'fit_slopes' method
    mx
        The value to use for the max of the peak. If None, np.amax(hist) is
        used.
    dmx
        The uncertainty in mx
    bl
        Used to specify an offset from which to estimate the FWFM.
    dbl
        The uncertainty in the bl
    method
        'bins_over_f' : the simplest method: just take the difference in the bin
            centers that are over [fraction] of max. Only works for high stats and
            FWFM/bin_width >> 1
        'interpolate' : interpolate between the bins that cross the [fraction]
            line.  Works well for high stats and a reasonable number of bins.
            Uncertainty incorporates var, if provided.
        'fit_slopes' : fit over n_slope bins in the vicinity of the FWFM and
            interpolate to get the fractional crossing point. Works okay even
            when stats are moderate but requires enough bins that dx traversed
            by n_slope bins is approximately linear. Incorporates bin variances
            in fit and final uncertainties if provided.
    n_slope
        Number of bins in the vicinity of the FWFM used to interpolate the fractional
        crossing point with the 'fit_slopes' method

    Returns
    -------
    fwfm, dfwfm
        fwfm: the full width at [fraction] of the maximum above bl
        dfwfm: the uncertainty in fwfm

    Examples
    --------
    >>> import pygama.analysis.histograms as pgh
    >>> from numpy.random import normal
    >>> hist, bins, var = pgh.get_hist(normal(size=10000), bins=100, range=(-5,5))
    >>> pgh.get_fwfm(0.5, hist, bins, var, method='bins_over_f')
    (2.2, 0.15919638684132664) # may vary

    >>> pgh.get_fwfm(0.5, hist, bins, var, method='interpolate')
    (2.2041666666666666, 0.09790931254396479) # may vary

    >>> pgh.get_fwfm(0.5, hist, bins, var, method='fit_slopes')
    (2.3083363869003466, 0.10939486522749278) # may vary
    """
    # find bins over [fraction]
    if mx is None:
        mx = np.amax(hist)
        if var is not None and dmx == 0:
            dmx = np.sqrt(var[np.argmax(hist)])
    idxs_over_f = hist > (bl + fraction * (mx - bl))

    # argmax will return the index of the first occurrence of a maximum
    # so we can use it to find the first and last time idxs_over_f is "True"
    bin_lo = np.argmax(idxs_over_f)
    bin_hi = len(idxs_over_f) - np.argmax(idxs_over_f[::-1])
    bin_centers = get_bin_centers(bins)

    # precalc dheight: uncertainty in height used as the threshold
    dheight2 = (fraction * dmx) ** 2 + ((1 - fraction) * dbl) ** 2

    if method == "bins_over_f":
        # the simplest method: just take the difference in the bin centers
        fwfm = bin_centers[bin_hi] - bin_centers[bin_lo]

        # compute rough uncertainty as [bin width] (+) [dheight / slope]
        dx = bin_centers[bin_lo] - bin_centers[bin_lo - 1]
        dy = hist[bin_lo] - hist[bin_lo - 1]
        if dy == 0:
            dy = (hist[bin_lo + 1] - hist[bin_lo - 2]) / 3
        dfwfm2 = dx**2 + dheight2 * (dx / dy) ** 2
        dx = bin_centers[bin_hi + 1] - bin_centers[bin_hi]
        dy = hist[bin_hi] - hist[bin_hi + 1]
        if dy == 0:
            dy = (hist[bin_hi - 1] - hist[bin_hi + 2]) / 3
        dfwfm2 += dx**2 + dheight2 * (dx / dy) ** 2
        return fwfm, np.sqrt(dfwfm2)

    elif method == "interpolate":
        # interpolate between the two bins that cross the [fraction] line
        # works well for high stats
        if bin_lo < 1 or bin_hi >= len(hist) - 1:
            raise ValueError(f"Can't interpolate ({bin_lo}, {bin_hi})")

        val_f = bl + fraction * (mx - bl)

        # x_lo
        dx = bin_centers[bin_lo] - bin_centers[bin_lo - 1]
        dhf = val_f - hist[bin_lo - 1]
        dh = hist[bin_lo] - hist[bin_lo - 1]
        x_lo = bin_centers[bin_lo - 1] + dx * dhf / dh
        # uncertainty
        dx2_lo = 0
        if var is not None:
            dx2_lo = (dhf / dh) ** 2 * var[bin_lo] + ((dh - dhf) / dh) ** 2 * var[
                bin_lo - 1
            ]
            dx2_lo *= (dx / dh) ** 2
        dd_dh = -dx / dh

        # x_hi
        dx = bin_centers[bin_hi + 1] - bin_centers[bin_hi]
        dhf = hist[bin_hi] - val_f
        dh = hist[bin_hi] - hist[bin_hi + 1]
        if dh == 0:
            raise ValueError("Interpolation failed, dh == 0")
        x_hi = bin_centers[bin_hi] + dx * dhf / dh
        if x_hi < x_lo:
            raise ValueError("Interpolation produced negative fwfm")
        # uncertainty
        dx2_hi = 0
        if var is not None:
            dx2_hi = (dhf / dh) ** 2 * var[bin_hi + 1] + ((dh - dhf) / dh) ** 2 * var[
                bin_hi
            ]
            dx2_hi *= (dx / dh) ** 2
        dd_dh += dx / dh

        return x_hi - x_lo, np.sqrt(dx2_lo + dx2_hi + dd_dh**2 * dheight2)

    elif method == "fit_slopes":
        # evaluate the [fraction] point on a line fit to n_slope bins near the crossing.
        # works okay even when stats are moderate
        val_f = bl + fraction * (mx - bl)

        # x_lo
        i_0 = bin_lo - int(np.floor(n_slope / 2))
        if i_0 < 0:
            raise RuntimeError("Fit slopes failed")
        i_n = i_0 + n_slope
        wts = (
            None if var is None else 1 / np.sqrt(var[i_0:i_n])
        )  # fails for any var = 0
        wts = [w if w != np.inf else 0 for w in wts]

        try:
            (m, b), cov = np.polyfit(
                bin_centers[i_0:i_n], hist[i_0:i_n], 1, w=wts, cov="unscaled"
            )
        except np.linalg.LinAlgError:
            raise RuntimeError("LinAlgError in x_lo")
        x_lo = (val_f - b) / m
        # uncertainty
        dxl2 = (
            cov[0, 0] / m**2
            + (cov[1, 1] + dheight2) / (val_f - b) ** 2
            + 2 * cov[0, 1] / (val_f - b) / m
        )
        dxl2 *= x_lo**2

        # x_hi
        i_0 = bin_hi - int(np.floor(n_slope / 2)) + 1
        if i_0 == len(hist):
            raise RuntimeError("Fit slopes failed")

        i_n = i_0 + n_slope
        wts = None if var is None else 1 / np.sqrt(var[i_0:i_n])
        wts = [w if w != np.inf else 0 for w in wts]
        try:
            (m, b), cov = np.polyfit(
                bin_centers[i_0:i_n], hist[i_0:i_n], 1, w=wts, cov="unscaled"
            )
        except np.linalg.LinAlgError:
            raise RuntimeError("LinAlgError in x_hi")
        x_hi = (val_f - b) / m
        if x_hi < x_lo:
            raise RuntimeError("Fit slopes produced negative fwfm")

        # uncertainty
        dxh2 = (
            cov[0, 0] / m**2
            + (cov[1, 1] + dheight2) / (val_f - b) ** 2
            + 2 * cov[0, 1] / (val_f - b) / m
        )
        dxh2 *= x_hi**2

        return x_hi - x_lo, np.sqrt(dxl2 + dxh2)

    else:
        raise NameError(f"Unrecognized method {method}")


def plot_hist(
    hist: np.ndarray,
    bins: np.ndarray,
    var: Optional[np.ndarray] = None,
    show_stats: bool = False,
    stats_hloc: float = 0.75,
    stats_vloc: float = 0.85,
    fill: bool = False,
    fillcolor: str = "r",
    fillalpha: float = 0.2,
    **kwargs,
) -> None:
    """
    Plot a step histogram, with optional error bars

    Parameters
    ----------
    hist
        The histogram data array containing the peak
    bins
        An array of bin edges for the histogram
    var
        An array of histogram variances. Used with the 'fit_slopes' method
    show_stats
        If True, shows the mean, mean error, and standard deviation of the histogram on the plot
    stats_hloc
        matplotlib.pyplot horizontal location to place the stats
    stats_vloc
        matplotlib.pyplot vertical location to place the stats
    fill
        If True, fills in a step histogram when plotting
    fill_color
        Color to fill the step histogram if fill is True
    fill_alpha
        Alpha amount if fill is True
    """
    if fill:
        # the concat calls get the steps to draw correctly at the range boundaries
        # where="post" tells plt to draw the step y[i] between x[i] and x[i+1]
        save_color = None
        if "color" in kwargs:
            save_color = kwargs.pop("color")
        elif "c" in kwargs:
            save_color = kwargs.pop("c")
        plt.fill_between(
            np.concatenate(([bins[0]], bins)),
            np.concatenate(([0], hist, [0])),
            step="post",
            color=fillcolor,
            alpha=fillalpha,
            **kwargs,
        )
        if save_color is not None:
            kwargs["color"] = save_color
    if var is None:
        # the concat calls get the steps to draw correctly at the range boundaries
        # where="post" tells plt to draw the step y[i] between x[i] and x[i+1]
        plt.step(
            np.concatenate(([bins[0]], bins)),
            np.concatenate(([0], hist, [0])),
            where="post",
            **kwargs,
        )
    else:
        plt.errorbar(
            get_bin_centers(bins),
            hist,
            xerr=get_bin_widths(bins) / 2,
            yerr=np.sqrt(var),
            fmt="none",
            **kwargs,
        )
    if show_stats is True:
        bin_centers = get_bin_centers(bins)
        n = np.sum(hist)
        if n <= 1:
            raise RuntimeError(f"can't compute sigma for n = {n}")
        mean = np.sum(hist * bin_centers) / n
        x2ave = np.sum(hist * bin_centers * bin_centers) / n
        stddev = np.sqrt(n / (n - 1) * (x2ave - mean * mean))
        dmean = stddev / np.sqrt(n)

        mean, dmean = pgu.get_formatted_stats(mean, dmean, 2)
        stats = f"$\\mu={mean} \\pm {dmean}$\n$\\sigma={stddev:#.3g}$"
        stats_fontsize = rcParams["legend.fontsize"]
        plt.text(
            stats_hloc,
            stats_vloc,
            stats,
            transform=plt.gca().transAxes,
            fontsize=stats_fontsize,
        )


def get_gaussian_guess(
    hist: np.ndarray, bins: np.ndarray
) -> tuple[float, float, float]:
    """
    given a hist, gives guesses for mu, sigma, and amplitude

    Parameters
    ----------
    hist
        Array of histogrammed data
    bins
        Array of histogram bins

    Returns
    -------
    guess_mu
        guess for the mu parameter of a Gaussian
    guess_sigma
        guess for the sigma parameter of a Gaussian
    guess_area
        guess for the A parameter of a Gaussian
    """
    if len(bins) == len(hist):
        log.warning(
            "note: this function has been updated to require bins rather than bin_centers. Don't trust this result"
        )

    max_idx = np.argmax(hist)
    guess_mu = (bins[max_idx] + bins[max_idx]) / 2  # bin center
    guess_amp = hist[max_idx]

    # find 50% amp bounds on both sides for a FWHM guess
    guess_sigma = get_fwhm(hist, bins)[0] / 2.355  # FWHM to sigma
    guess_area = guess_amp * guess_sigma * np.sqrt(2 * np.pi)

    return (guess_mu, guess_sigma, guess_area)


def get_bin_estimates(
    pars: np.ndarray,
    func: Callable,
    bins: np.ndarray,
    is_integral: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    Bin expected means are estimated by f(bin_center)*bin_width. Supply an
    integrating function to compute the integral over the bin instead.
    TODO: make default integrating function a numerical method that is off by
    default.

    Parameters
    ----------
    pars
        The parameters of the function, func
    func
        The function at which to evaluate the bin centers
    bins
        Array of histogram bins
    is_integral
        if True, then func is an integral function

    Returns
    -------
    f(bin_center)*bin_width
        The expected mean of a bin

    See Also
    --------
    get_bin_widths
        Returns the width of the bins
    get_bin_centers
        Finds the bin centers of the supplied bins
    """
    if is_integral is False:
        return func(get_bin_centers(bins), *pars, **kwargs) * get_bin_widths(bins)
    else:
        # func can be an integral functions
        return func(bins[1:], *pars, **kwargs) - func(bins[:-1], *pars, **kwargs)


def get_i_local_extrema(data, delta):
    """Get lists of indices of the local maxima and minima of data

    The "local" extrema are those maxima / minima that have heights / depths of
    at least delta.
    Converted from MATLAB script at: http://billauer.co.il/peakdet.html

    Parameters
    ----------
    data : array-like
        the array of data within which extrema will be found
    delta : scalar
        the absolute level by which data must vary (in one direction) about an
        extremum in order for it to be tagged

    Returns
    -------
    imaxes, imins : 2-tuple ( array, array )
        A 2-tuple containing arrays of variable length that hold the indices of
        the identified local maxima (first tuple element) and minima (second
        tuple element)
    """

    # prepare output
    imaxes, imins = [], []

    # sanity checks
    data = np.asarray(data)
    if not np.isscalar(delta):
        log.error("get_i_local_extrema: Input argument delta must be a scalar")
        return np.array(imaxes), np.array(imins)
    if delta <= 0:
        log.error(f"get_i_local_extrema: delta ({delta}) must be positive")
        return np.array(imaxes), np.array(imins)

    # now loop over data
    imax, imin = 0, 0
    find_max = True
    for i in range(len(data)):
        if data[i] > data[imax]:
            imax = i
        if data[i] < data[imin]:
            imin = i

        if find_max:
            # if the sample is less than the current max by more than delta,
            # declare the previous one a maximum, then set this as the new "min"
            if data[i] < data[imax] - delta:
                imaxes.append(imax)
                imin = i
                find_max = False
        else:
            # if the sample is more than the current min by more than delta,
            # declare the previous one a minimum, then set this as the new "max"
            if data[i] > data[imin] + delta:
                imins.append(imin)
                imax = i
                find_max = True

    return np.array(imaxes), np.array(imins)


def get_i_local_maxima(data, delta):
    return get_i_local_extrema(data, delta)[0]


def get_i_local_minima(data, delta):
    return get_i_local_extrema(data, delta)[1]
