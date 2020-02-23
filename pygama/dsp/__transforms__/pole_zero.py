import numpy as np
from numba import guvectorize


@guvectorize(["void(float32[:], float32, float32[:])",
              "void(float64[:], float64, float64[:])"],
             "(n),()->(n)", forceobj=True, cache=True)
def pole_zero(wf_in, tau, wf_out):
    """
    Pole-zero correction using time constant tau
    """
    np.copy_to(wf_out, wf_in)
    np.multiply(wf_out, const, wf_out)
    np.subtract(wf_in[1::-1], wf_out[:-1:-1], wf_out[1::-1])
    np.copy_to(wf_out[0], wf_in[0])
    np.cumsum(wf_out, out=wf_out, axis=1)
