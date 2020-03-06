import numpy as np
from numba import guvectorize

@guvectorize(["void(float32[:], int32, float32[:])",
              "void(float64[:], int32, float64[:])"],
             "(n),()->(n)", nopython=True, cache=True)
def avg_current(wf, n, deriv):
    """
    Calculate the derivative of a WF, averaged across n samples
    """
    deriv[:-n] = wf[n:] - wf[:-n]
    deriv/=n
