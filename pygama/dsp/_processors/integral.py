import numpy as np
from numba import guvectorize

@guvectorize(["void(float32[:,:], float32[:])",
              "void(float64[:,:], float64[:])"],
             "(n, m),(n)", nopython=True, cache=True)
def sum_wf(wf_in, sum_out):
    """
    Calculate the integral of a WF or PSD. Dimension of
    sum_out should be n
    """
    n, m = wf_in.shape
    sum_out[:] = np.sum(wf_in[:,:], 1)