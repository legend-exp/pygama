# 1) Import Python modules

import numpy as np
from numba import guvectorize
from pygama.dsp.errors import DSPFatal

# 2) Provide instructions to Numba

@guvectorize(["void(float32[:], float32, float32[:], float32[:])",
              "void(float64[:], float64, float64[:], float64[:])"],
              "(n),(),(m)->()", nopython=True, cache=True)

# 3) Define the processor interface

def sum_at_positions(w_in, thresh_in, pos_in, sum_out):

    # 4) Document the algorithm
    #
    """
    Get the sum of the values of the input waveform at specified positions

    Parameters
    ----------
    w_in : array-like
        The array of data which specific values should be summed
    pos_in : array-like
        list of indices which indicate the location where the values of w_in should be summed
    thresh_in: scalar
        a simple threshold over which the value has to be to be included in the sum
    sum_out : scalar
       Provides sum of the specified values
    """

    # 5) Initialize output parameters

    sum_out[0] = np.nan # use [0] for scalar parameters

    # 6) Check inputs
   
    if np.isnan(w_in).any():
        return

    # 7) Algorithm
   
    sum_out[0] = 0.
    for i in range(0, len(pos_in), 1):
        if not np.isnan(pos_in[i]) and w_in[int(pos_in[i])]>thresh_in:
            sum_out[0] = sum_out[0] + w_in[int(pos_in[i])]
        else:
            return