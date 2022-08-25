import numpy as np
from numba import guvectorize


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:])",
    ],
    "(n),(m),(m)",
    nopython=True,
    cache=True,
)
def get_multi_time_pickoff(w_in, t_in, a_out):
    """
    Parameters
    ----------
    w_in : array-like
        The array of data within which extrema will be found
    t_in :  array-like
        the waveform indecies to pick off
    a_out,  array-like
        the output pick-off value
    """

    # prepare output

    a_out[:] = np.nan

    # fill output
    for i in range(len(t_in)):
        if t_in[i] != np.nan and t_in[i] >= 0 and t_in[i] < len(w_in):
            a_out[i] = w_in[int(t_in[i])]
