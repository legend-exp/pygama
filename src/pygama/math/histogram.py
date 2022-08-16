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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

import pygama.math.utils as pgu


def get_hist(data, bins=None, range=None, dx=None, wts=None):
    """return hist, bins, var after binning data

    This is just a wrapper for numpy.histogram, with optional weights for each
    element and proper computing of variances.

    Note: there are no overflow / underflow bins.

    Available binning methods:

      - Default (no binning arguments) : 100 bins over an auto-detected range
      - bins=N, range=(x_lo, x_hi) : N bins over the specified range (or leave
        range=None for auto-detected range)
      - bins=[str] : use one of np.histogram's automatic binning algorithms
      - bins=bin_edges_array : array lower bin edges, supports non-uniform binning
      - dx=dx, range=(x_lo, x_hi): bins of width dx over the specified range.
        Note: dx overrides the bins argument!

    Parameters
    ----------
    data : array like
        The array of data to be histogrammed
    bins : int, array, or str (optional)
        int: the number of bins to be used in the histogram
        array: an array of bin edges to use
        str: the name of the np.histogram automatic binning algorithm to use
        If not provided, np.histogram's default auto-binning routine is used
    range : tuple (float, float) (optional)
        (x_lo, x_high) is the tuple of low and high x values to uses for the
        very ends of the bin range. If not provided, np.histogram chooses the
        ends based on the data in data
    dx : float (optional)
        Specifies the bin width. Overrides "bins" if both arguments are present
    wts : float or array like (optional)
        Array of weights for each bin. For example, if you want to divide all
        bins by a time T to get the bin contents in count rate, set wts = 1/T.
        Variances will be computed for each bin that appropriately account for
        each data point's weighting.

    Returns
    -------
    hist : array
        the values in each bin of the histogram
    bins : array
        an array of bin edges: bins[i] is the lower edge of the ith bin.
        Note: it includes the upper edge of the last bin and does not
        include underflow or overflow bins. So len(bins) = len(hist) + 1
    var : array
        array of variances in each bin of the histogram
    """
    if bins is None:
        bins = 100 # override np.histogram default of just 10

    if dx is not None:
        bins = int((range[1] - range[0]) / dx)

    # bins includes left edge of first bin and right edge of all other bins
    # allow scalar weights
    if wts is not None and np.shape(wts) == (): wts = np.full_like(data, wts)
    hist, bins = np.histogram(data, bins=bins, range=range, weights=wts)

    if wts is None:
        # no weights: var = hist, but return a copy so that mods to var don't
        # modify hist.
        # Note: If you don't want a var copy, just call np.histogram()
        return hist, bins, hist.copy()
    else:
        # get the variances by binning with double the weight
        var, bins = np.histogram(data, bins=bins, weights=wts*wts)
        return hist, bins, var


def better_int_binning(x_lo=0, x_hi=None, dx=None, n_bins=None):
    """ Get a good binning for integer data.

    Guarantees an integer bin width.

    At least two of x_hi, dx, or n_bins must be provided.

    Parameters
    ----------
    x_lo : float
        Desired low x value for the binning
    x_hi : float
        Desired high x value for the binning
    dx : float
        Desired bin width
    n_bins : float
        Desired number of bins

    Returns
    -------
    x_lo: int
        int values for best x_lo
    x_hi: int
        int values for best x_hi, returned if x_hi is not None
    dx : int
        best int bin width, returned if arg dx is not None
    n_bins : int
        best int n_bins, returned if arg n_bins is not None
    """
    # process inputs
    n_Nones = int(x_hi is None) + int(dx is None) + int(n_bins is None)
    if n_Nones > 1:
        print('better_int_binning: must provide two of x_hi, dx or n_bins')
        return
    if n_Nones == 0:
        print('better_int_binning: overconstrained. Ignoring x_hi.')
        x_hi = None

    # get valid dx or n_bins
    if dx is not None:
        if dx <= 0:
            print(f'better_int_binning: invalid dx={dx}')
            return
        dx = np.round(dx)
        if dx == 0: dx = 1
    if n_bins is not None:
        if n_bins <= 0:
            print(f'better_int_binning: invalid n_bins={n_bins}')
            return
        n_bins = np.round(n_bins)

    # can already return if no x_hi
    if x_hi is None: # must have both dx and n_bins
        return int(x_lo), int(dx), int(n_bins)

    # x_hi is valid. Get a valid dx if we don't have one
    if dx is None: # must have n_bins
        dx = np.round((x_hi-x_lo)/n_bins)
    if dx == 0: dx = 1

    # Finally, build a good binning from dx
    n_bins = np.ceil((x_hi-x_lo)/dx)
    x_lo = np.floor(x_lo)
    x_hi = x_lo + n_bins*dx
    if n_bins is None: return int(x_lo), int(x_hi), int(dx)
    else: return int(x_lo), int(x_hi), int(n_bins)


def get_bin_centers(bins):
    """
    Returns an array of bin centers from an input array of bin edges.
    Works for non-uniform binning. Note: a new array is allocated

    Parameters:
    """
    return (bins[:-1] + bins[1:]) / 2.


def get_bin_widths(bins):
    """
    Returns an array of bin widths from an input array of bin edges.
    Works for non-uniform binning.
    """
    return (bins[1:] - bins[:-1])


def find_bin(x, bins):
    """
    Returns the index of the bin containing x
    Returns -1 for underflow, and len(bins) for overflow
    For uniform bins, jumps directly to the appropriate index.
    For non-uniform bins, binary search is used.
    """
    # first handle overflow / underflow
    if len(bins) == 0: return 0 # i.e. overflow
    if x < bins[0]: return -1
    if x > bins[-1]: return len(bins)

    # we are definitely in range, and there are at least 2 bin edges, one below
    # and one above x. try assuming uniform bins
    dx = bins[1]-bins[0]
    index = int(np.floor((x-bins[0])/dx))
    if bins[index] <= x and bins[index+1] > x: return index

    # bins are non-uniform: find by binary search
    return np.searchsorted(bins, x, side='right')


def range_slice(x_min, x_max, hist, bins, var=None):
    i_min = find_bin(x_min, bins)
    i_max = find_bin(x_max, bins)
    if var is not None: var = var[i_min:i_max]
    return hist[i_min:i_max], bins[i_min:i_max+1], var


def get_fwhm(hist, bins, var=None, mx=None, dmx=0, bl=0, dbl=0, method='bins_over_f', n_slope=3):
    """Estimate the FWHM of data in a histogram

    See Also
    --------
    get_fwfm : for parameters and return values
    """
    if len(bins) == len(hist):
        print("note: this function has been updated to require bins rather",
              "than bin_centers. Don't trust this result")
    return get_fwfm(0.5, hist, bins, var, mx, dmx, bl, dbl, method, n_slope)


def get_fwfm(fraction, hist, bins, var=None, mx=None, dmx=0, bl=0, dbl=0, method='bins_over_f', n_slope=3):
    """
    Estimate the full width at some fraction of the max of data in a histogram

    Typically used by sending slices around a peak. Searches about argmax(hist)
    for the peak to fall by [fraction] from mx to bl

    Parameters
    ----------
    fraction : float
        The fractional amplitude at which to evaluate the full width
    hist : array-like
        The histogram data array containing the peak
    bins : array-like
        An array of bin edges for the histogram
    var : array-like (optional)
        An array of histogram variances. Used with the 'fit_slopes' method
    mx : float or tuple(float, float) (optional)
        The value to use for the max of the peak. If None, np.amax(hist) is
        used.
    dmx : float (optional)
        The uncertainty in mx
    bl : float or tuple (float, float) (optional)
        Used to specify an offset from which to estimate the FWFM.
    dbl : float (optional)
        The uncertainty in the bl
    method : string
        'bins_over_f' : the simplest method: just take the difference in the bin
            centers that are over [fraction] of max. Only works for high stats and
            FWFM/bin_width >> 1
        'interpolate' : interpolate between the bins that cross the [fration]
            line.  Works well for high stats and a reasonable number of bins.
            Uncertainty incorporates var, if provided.
        'fit_slopes' : fit over n_slope bins in the vicinity of the FWFM and
            interpolate to get the fractional crossing point. Works okay even
            when stats are moderate but requires enough bins that dx traversed
            by n_slope bins is approximately linear. Incorporates bin variances
            in fit and final uncertainties if provided.
    n_slope : int
        DOCME

    Returns
    -------
    fwfm, dfwfm : float, float
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
    idxs_over_f = hist > (bl + fraction * (mx-bl))

    # argmax will return the index of the first occurrence of a maximum
    # so we can use it to find the first and last time idxs_over_f is "True"
    bin_lo = np.argmax(idxs_over_f)
    bin_hi = len(idxs_over_f) - np.argmax(idxs_over_f[::-1])
    bin_centers = get_bin_centers(bins)

    # precalc dheight: uncertainty in height used as the threshold
    dheight2 = (fraction*dmx)**2 + ((1-fraction)*dbl)**2

    if method == 'bins_over_f':
        # the simplest method: just take the difference in the bin centers
        fwfm = bin_centers[bin_hi] - bin_centers[bin_lo]

        # compute rough uncertainty as [bin width] (+) [dheight / slope]
        dx = bin_centers[bin_lo] - bin_centers[bin_lo-1]
        dy = hist[bin_lo] - hist[bin_lo-1]
        if dy == 0: dy = (hist[bin_lo+1] - hist[bin_lo-2])/3
        dfwfm2 = dx**2 + dheight2 * (dx/dy)**2
        dx = bin_centers[bin_hi+1] - bin_centers[bin_hi]
        dy = hist[bin_hi] - hist[bin_hi+1]
        if dy == 0: dy = (hist[bin_hi-1] - hist[bin_hi+2])/3
        dfwfm2 += dx**2 + dheight2 * (dx/dy)**2
        return fwfm, np.sqrt(dfwfm2)

    elif method == 'interpolate':
        # interpolate between the two bins that cross the [fraction] line
        # works well for high stats
        if bin_lo < 1 or bin_hi >= len(hist)-1:
            print(f"get_fwhm: can't interpolate ({bin_lo}, {bin_hi})")
            return 0, 0

        val_f = bl + fraction*(mx-bl)

        # x_lo
        dx = bin_centers[bin_lo] - bin_centers[bin_lo-1]
        dhf = val_f - hist[bin_lo-1]
        dh = hist[bin_lo] - hist[bin_lo-1]
        x_lo = bin_centers[bin_lo-1] + dx * dhf/dh
        # uncertainty
        dx2_lo = 0
        if var is not None:
            dx2_lo = (dhf/dh)**2 * var[bin_lo] + ((dh-dhf)/dh)**2 * var[bin_lo-1]
            dx2_lo *= (dx/dh)**2
        dDdh = -dx/dh

        # x_hi
        dx = bin_centers[bin_hi+1] - bin_centers[bin_hi]
        dhf = hist[bin_hi] - val_f
        dh = hist[bin_hi] - hist[bin_hi+1]
        if dh == 0:
            raise ValueError(f"get_fwhm: interpolation failed, dh == 0")
        x_hi = bin_centers[bin_hi] + dx * dhf/dh
        if x_hi < x_lo:
            raise ValueError(f"get_fwfm: interpolation produced negative fwfm")
        # uncertainty
        dx2_hi = 0
        if var is not None:
            dx2_hi = (dhf/dh)**2 * var[bin_hi+1] + ((dh-dhf)/dh)**2 * var[bin_hi]
            dx2_hi *= (dx/dh)**2
        dDdh += dx/dh

        return x_hi - x_lo, np.sqrt(dx2_lo + dx2_hi + dDdh**2 * dheight2)

    elif method == 'fit_slopes':
        # evaluate the [fraction] point on a line fit to n_slope bins near the crossing.
        # works okay even when stats are moderate
        val_f = bl + fraction*(mx-bl)

        # x_lo
        i_0 = bin_lo - int(np.floor(n_slope/2))
        if i_0 < 0:
            print(f"get_fwfm: fit slopes failed")
            return 0, 0
        i_n = i_0 + n_slope
        wts = None if var is None else 1/np.sqrt(var[i_0:i_n]) #fails for any var = 0
        wts = [w if w != np.inf else 0 for w in wts]

        try:
            (m, b), cov = np.polyfit(bin_centers[i_0:i_n], hist[i_0:i_n], 1, w=wts, cov='unscaled')
        except np.linalg.LinAlgError:
            print(f"get_fwfm: LinAlgError")
            return 0, 0
        x_lo = (val_f-b)/m
        #uncertainty
        dxl2 = cov[0,0]/m**2 + (cov[1,1] + dheight2)/(val_f-b)**2 + 2*cov[0,1]/(val_f-b)/m
        dxl2 *= x_lo**2

        # x_hi
        i_0 = bin_hi - int(np.floor(n_slope/2)) + 1
        if i_0 == len(hist):
            print(f"get_fwfm: fit slopes failed")
            return 0, 0

        i_n = i_0 + n_slope
        wts = None if var is None else 1/np.sqrt(var[i_0:i_n])
        wts = [w if w != np.inf else 0 for w in wts]
        try:
            (m, b), cov = np.polyfit(bin_centers[i_0:i_n], hist[i_0:i_n], 1, w=wts, cov='unscaled')
        except np.linalg.LinAlgError:
            print(f"get_fwfm: LinAlgError")
            return 0, 0
        x_hi = (val_f-b)/m
        if x_hi < x_lo:
            print(f"get_fwfm: fit slopes produced negative fwfm")
            return 0, 0

        #uncertainty
        dxh2 = cov[0,0]/m**2 + (cov[1,1] + dheight2)/(val_f-b)**2 + 2*cov[0,1]/(val_f-b)/m
        dxh2 *= x_hi**2

        return x_hi - x_lo, np.sqrt(dxl2 + dxh2)

    else:
        print(f"get_fwhm: unrecognized method {method}")
        return 0, 0


def plot_hist(hist, bins, var=None, show_stats=False, stats_hloc=0.75, stats_vloc=0.85, fill=False, fillcolor='r', fillalpha=0.2, **kwargs):
    """
    plot a step histogram, with optional error bars
    """
    if fill:
        # the concat calls get the steps to draw correctly at the range boundaries
        # where="post" tells plt to draw the step y[i] between x[i] and x[i+1]
        save_color = None
        if 'color' in kwargs: save_color = kwargs.pop('color')
        elif 'c' in kwargs: save_color = kwargs.pop('c')
        plt.fill_between(np.concatenate(([bins[0]], bins)), np.concatenate(([0], hist, [0])), step="post", color=fillcolor, alpha=fillalpha, **kwargs)
        if save_color is not None: kwargs['color'] = save_color
    if var is None:
        # the concat calls get the steps to draw correctly at the range boundaries
        # where="post" tells plt to draw the step y[i] between x[i] and x[i+1]
        plt.step(np.concatenate(([bins[0]], bins)), np.concatenate(([0], hist, [0])), where="post", **kwargs)
    else:
        plt.errorbar(get_bin_centers(bins), hist,
                     xerr=get_bin_widths(bins) / 2, yerr=np.sqrt(var),
                     fmt='none', **kwargs)
    if show_stats is True:
        bin_centers = get_bin_centers(bins)
        N = np.sum(hist)
        if N <= 1:
            print("can't compute sigma for N =", N)
            return
        mean = np.sum(hist*bin_centers)/N
        x2ave = np.sum(hist*bin_centers*bin_centers)/N
        stddev = np.sqrt(N/(N-1) * (x2ave - mean*mean))
        dmean = stddev/np.sqrt(N)

        mean, dmean = pgu.get_formatted_stats(mean, dmean, 2)
        stats = f'$\\mu={mean} \\pm {dmean}$\n$\\sigma={stddev:#.3g}$'
        stats_fontsize = rcParams['legend.fontsize']
        plt.text(stats_hloc, stats_vloc, stats, transform=plt.gca().transAxes, fontsize = stats_fontsize)


def get_gaussian_guess(hist, bins):
    """
    given a hist, gives guesses for mu, sigma, and amplitude
    """
    if len(bins) == len(hist):
        print("note: this function has been updated to require bins rather",
              "than bin_centers. Don't trust this result")

    max_idx = np.argmax(hist)
    guess_e = (bins[max_idx] + bins[max_idx])/2 # bin center
    guess_amp = hist[max_idx]

    # find 50% amp bounds on both sides for a FWHM guess
    guess_sigma = get_fwhm(hist, bins)[0] / 2.355  # FWHM to sigma
    guess_area = guess_amp * guess_sigma * np.sqrt(2 * np.pi)

    return (guess_e, guess_sigma, guess_area)
