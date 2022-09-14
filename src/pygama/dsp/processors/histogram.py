from __future__ import annotations

from math import floor

import numpy as np
from numba import guvectorize

from pygama.dsp.utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:], float64[:])",
    ],
    "(n),(m),(p),(q)",
    **nb_kwargs,
)
def histogram(
    w_in: np.ndarray, widths_in: np.ndarray, weights_out: np.ndarray, borders_out: np.ndarray
) -> None:

    """Produces and returns an histogram of the waveform.

    Parameters
    ----------
    w_in
        Data to be histogrammed.
    widths_in
        Widths of each bin. If only 1 element is given, equal widths are assumed and the number of bins  equals the only element in widths_in
    weights_out
        The output histogram weights.
    borders_out
        The output histogram bin edges of the histogram.

    Note
    ----
    This implementation is significantly faster than just wrapping
    :func:`numpy.histogram`.

    See Also
    --------
    .histogram_stats
    """

    weights_out[:] = 0
    borders_out[:] = np.nan

    if np.isnan(w_in).any() or np.isnan(widths_in).any():
        return

    #create the bin borders
    borders_out[0] = min(w_in)
    delta=0
    
    #in case bin widths are given
    if len(widths_in) > 1:
        delta = (max(w_in)-min(w_in))/np.sum(widths_in)
        for i in range(0,len(widths_in),1):
            borders_out[i+1] = min(w_in) + sum(widths_in[0:i+1])*delta

    #else equal spaced bins
    else:
        # number of bins
        bin_in = widths_in[0]

        # define the bin edges
        delta = (max(w_in) - min(w_in)) / (bin_in)
        for i in range(0, bin_in, 1):
            borders_out[i+1] = min(w_in) + delta *(i+1)

    
    # make the histogram
    for i in range(0, len(w_in), 1):
        for k in range(1, len(borders_out), 1):
            if (w_in[i] - borders_out[k]) < 0:
                weights_out[k-1] += 1
                break


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:], float32[:],float32[:],float32)",
        "void(float64[:], float64[:], float64[:], float64[:],float64[:],float64)",
    ],
    "(n),(m),(),(),(),()",
    **nb_kwargs,
)
def histogram_stats(
    weights_in: np.ndarray,
    edges_in: np.ndarray,
    mode_out: int,
    max_out: float,
    fwhm_out: float,
    max_in: float,
) -> None:

    """Compute useful histogram-related quantities.

    Parameters
    ----------
    weights_in
        histogram weights.
    edges_in
        histogram bin edges.
    max_in
        if not :any:`numpy.inf`, this value is used as the histogram bin
        content at the mode.  Otherwise the mode is computed automatically
        (from left to right).
    mode_out
        the computed mode of the histogram. If `max_in` is not :any:`numpy.inf`
        then the closest waveform index to `max_in` is returned.
    max_out
        the histogram bin content at the mode.
    fwhm_out
        the FWHM of the histogram, calculated by starting from the mode and
        descending left and right.

    See Also
    --------
    .histogram
    """

    fwhm_out[0] = np.nan
    mode_out[0] = np.nan
    max_out[0] = np.nan

    if np.isnan(weights_in).any():
        return

    # find global maximum search from left to right
    max_index = 0
    if np.isnan(max_in):
        for i in range(0, len(weights_in), 1):
            if weights_in[i] > weights_in[max_index]:
                max_index = i

    # is user specifies mean justfind mean index
    else:
        if (max_in>edges_in[-2]):
            max_index = len(weights_in) - 1
        else:
            for i in range(0, len(weights_in), 1):
                if abs(max_in - edges_in[i]) < abs(max_in - edges_in[max_index]):
                    max_index = i

    mode_out[0] = max_index
    #returns left bin edge
    max_out[0] = edges_in[max_index]

    # and the approx fwhm
    for i in range(max_index, len(weights_in), 1):
        if weights_in[i] <= 0.5 * weights_in[max_index] and weights_in[i] != 0:
            fwhm_out[0] = abs(max_out[0] - edges_in[i])
            break

    # look also into the other direction
    for i in range(0, max_index, 1):
        if weights_in[i] >= 0.5 * weights_in[max_index] and weights_in[i] != 0:
            if fwhm_out[0] < abs(max_out[0] - edges_in[i]):
                fwhm_out[0] = abs(max_out[0] - edges_in[i])
            break
