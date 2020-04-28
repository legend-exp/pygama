import numpy as np
from numba import guvectorize

@guvectorize(["void(float32[:], int32, float32[:])",
              "void(float64[:], int32, float64[:])"],
             "(n),(),(m)", nopython=True, cache=True)
def avg_current(wf, n, deriv):
    """
    Calculate the derivative of a WF, averaged across n samples. Dimension of
    deriv should be len(wf) - n
    """
    deriv[:] = wf[n:] - wf[:-n]
    deriv/=n
