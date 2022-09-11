from __future__ import annotations

import numpy as np
from numba import guvectorize


@guvectorize(
    [
        "void(float32[:], float32[:], float32, float32, float32[:], float32[:])",
        "void(float64[:], float64[:], float64, float64, float64[:], float64[:])",
    ],
    "(n),(m),(),(),(m),()",
    nopython=True,
    cache=True,
)
def peak_snr_threshold(
    w_in: np.ndarray,
    idx_in: np.ndarray,
    ratio_in: float,
    width_in: int,
    idx_out: np.ndarray,
    n_idx_out: int,
) -> None:
    """
    Searches for local minima in a window consisting of +- the given witdth around the provided indices. If a minima is found it is checked if the amplitude of the minima  divided by the amplitude of the waveform at index is smaller then the given ratio. If this is the case the index is passed to the output.

    Parameters
    ----------
    w_in : array-like
        The array of data within which noise will be found
    idx_in : array-like
        The array of indices of possible signal candidates
    ratio_in :  float
        noise cancel sensitivity
    width_in: int
        width about index to analyse for noise
    idx_out,  array-like
        the cleaned inex array
    n_idx_out,  int
        Number of indices in idx_out with a non nan value
    """

    # prepare output

    idx_out[:] = np.nan
    n_idx_out[0] = 0

    # fill output
    k = 0
    for i in range(len(idx_in)):
        if not np.isnan(idx_in[i]):
            a = int(idx_in[i]) - int(width_in)
            b = int(idx_in[i]) + int(width_in)
            if a < 0:
                a = 0
            if b >= len(w_in):
                b = len(w_in) - 1
            min_index = a
            for j in range(a, b, 1):
                if w_in[j] < w_in[min_index]:
                    min_index = j
            if np.absolute(w_in[min_index] / w_in[int(idx_in[i])]) < ratio_in:
                idx_out[k] = idx_in[i]
                k = k + 1
    n_idx_out[0] = k
