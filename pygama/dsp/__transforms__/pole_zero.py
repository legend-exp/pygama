import numpy as np
from numba import guvectorize


@guvectorize(["void(float32[:], float32, float32[:])",
              "void(float64[:], float64, float64[:])"],
             "(n),()->(n)", forceobj=True, cache=True)
def pole_zero(wf_in, tau, wf_out):
    """
    Pole-zero correction using time constant tau
    """
    # Do the following all in place within wf_in and wf_out:
    # wf_out[i] = wf_out[i-1] + wf_in[i] - e^-1/tau*wf_in[i-1]
    const = np.exp(-1/tau)
    np.copyto(wf_out, wf_in)
    np.multiply(wf_out, const, wf_out)
    np.subtract(wf_in[-1:1:-1], wf_out[-2:0:-1], wf_out[-1:1:-1])
    np.cumsum(wf_out, out=wf_out, axis=0)
