# 1) Import Python modules

from math import floor

import numpy as np
from numba import guvectorize

from pygama.dsp.errors import DSPFatal

# 2) Provide instructions to Numba


@guvectorize(
    [
        "void(float32[:], float32, float32[:], float32[:])",
        "void(float64[:], float64, float64[:], float64[:])",
    ],
    "(n),(),(m),(m)",
    nopython=True,
    cache=True,
)

# 3) Define the processor interface


def y_projection(w_in, bin_in, proj_out, borders_out):

    # 4) Document the algorithm

    """
    Produces and returns a binned projection onto the y-axis from a provided waveform

    Parameters
    ----------
    w_in : array-like
        The array of data within which should be projected
    bin_in : scalar
        The number of bins
    proj_out : array-like
        Returns the projection as a binned array
    borders_out : array-like
        Returns the bin edges of the projection
    """

    # 5) Initialize output parameters

    proj_out[:] = 0
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
        proj_out[j] += 1
    # h, b= np.histogram(w_in, np.linspace(0,bin_in,1))
