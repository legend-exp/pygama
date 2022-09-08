from __future__ import annotations
from cmath import isnan

from math import floor

import numpy as np
from numba import guvectorize

from pygama.dsp.errors import DSPFatal
from pygama.dsp.utils import numba_defaults_kwargs as nb_kwargs

@guvectorize(
    [
        "void(float32[:], float32, float32[:], float32[:])",
        "void(float64[:], float64, float64[:], float64[:])",
    ],
    "(n),(),(m),(m)",
    **nb_kwargs
)
def histogram(w_in: np.ndarray, bin_in: int, weights_out: np.ndarray, borders_out: np.ndarray) -> None:

    """
    Produces and returns a binned histogram from a provided waveform

    Note
    ----
    Faster then wrapping numpy.histogram(). 

    Parameters
    ----------
    w_in : array-like
        The array of data within which should be projected

    bin_in : int
        The number of bins

    weights_out : array-like
        Returns the histogram weights of the input waveform

    borders_out : array-like
        Returns the bin edges of the histogram
    """

    # 5) Initialize output parameters

    weights_out[:] = 0
    borders_out[:] = np.nan

    # 6) Check inputs

    if np.isnan(w_in).any() or np.isnan(bin_in):
        return

    if bin_in < 1:
        raise DSPFatal("Bin size must be >=1")

    # 7) Algorithm

    # define our bin edges
    delta = (max(w_in) - min(w_in)) / (bin_in - 1)

    for i in range(0, bin_in, 1):
        borders_out[i] = min(w_in) + delta * i

    # get the projection on the y axis
    for i in range(0, len(w_in), 1):
        j = floor((w_in[i] - borders_out[0]) / delta)
        if j < 0:
            j = 0
        if j >= len(borders_out):
            j = len(borders_out) - 1
        weights_out[j] += 1

@guvectorize(
    [
        "void(float32[:], float32[:], float32[:], float32[:],float32[:],float32)",
        "void(float64[:], float64[:], float64[:], float64[:],float64[:],float64)",
    ],
    "(n),(n),(),(),(),()",
        **nb_kwargs
)
def histogram_stats(weights_in: np.ndarray, edges_in: np.ndarray, mode_out: int, max_out: float, fwhm_out: float, max_in:float) -> None:

    """
    Provided a projection of a waveform onto the y axis, the baseline is reconstructed by assuming mean of 0 of the projection and the stddev from it.

    Parameters
    ----------
    weights_in : array-like
        Weights of a binned histogram

    edges_in : array-like
        The bin borders of the histogram

    max_in : float, Optional
        If passed, this value is used as the max. of the histogram. If not the maximum of the histogram is search automatically (from left to right).

    mode_out : int
        Returns the index of the maximum of the histogram. If max_in is passed the closest index to the input maximum is returned.
    
    max_out : float
        Returns the the maximum of the histogram.

    fwhm_out : float
        Returns the FWHM of the the histogram. 
    """

    # 5) Initialize output parameters

    fwhm_out[0] = np.nan
    mode_out[0] = np.nan
    max_out[0]  = np.nan

    # 6) Check inputs

    if np.isnan(weights_in).any():
        return

    # 7) Algorithm

    # find global maximum search from left to right
    max_index = 0
    if np.isnan(max_in):
        for i in range(0, len(weights_in), 1):
            if weights_in[i] > weights_in[max_index]:
                max_index = i       

    # is user specifies mean justfind mean index
    else:
        for i in range(0, len(weights_in), 1):
            if abs(max_in - edges_in[i]) < abs(max_in - edges_in[max_index]):
                max_index = i
    
    mode_out[0] = max_index
    max_out[0] = edges_in[max_index]

    # and the approx standarddev
    for i in range(max_index, len(weights_in) - 1, 1):
        if weights_in[i] <= 0.5 * weights_in[max_index] and weights_in[i] != 0:
            fwhm_out[0] = abs(max_out[0] - edges_in[i])
            break

    # look also into the other direction
    for i in range(1, max_index, 1):
        if weights_in[i] >= 0.5 * weights_in[max_index] and weights_in[i] != 0:
            if fwhm_out[0] < abs(max_out[0] - edges_in[i]):
                fwhm_out[0] = abs(max_out[0] - edges_in[i])
            break
