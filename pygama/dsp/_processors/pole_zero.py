import numpy as np
from numba import guvectorize


@guvectorize(["void(float32[:], float32, float32[:])",
              "void(float64[:], float64, float64[:])"],
             "(n),()->(n)", nopython=True, cache=True)
def pole_zero(wf_in, tau, wf_out):
    """
    Pole-zero correction using time constant tau
    """
    const = np.exp(-1/tau)
    wf_out[0] = wf_in[0]
    for i in range(1, len(wf_in)):
        wf_out[i] = wf_out[i-1] + wf_in[i] - wf_in[i-1]*const

@guvectorize(["void(float32[:], float32, float32, float32, float32[:])",
              "void(float64[:], float64, float64, float64, float64[:])"],
             "(n),(),(),()->(n)", nopython=True, cache=True)
def double_pole_zero(wf_in, tau1, tau2, frac, wf_out):
    """
    Pole-zero correction using two time constants: one main (long) time constant
    tau1, and a shorter time constant tau2 that contributes a fraction frac
    """
    const1 = 1/tau1 #np.exp(-1/tau1)
    const2 = 1/tau2 #np.exp(-1/tau2)
    wf_out[0] = wf_in[0]
    e1 = wf_in[0]
    e2 = wf_in[0]
    e3 = 0
    for i in range(1, len(wf_in)):
        e1 += wf_in[i] - e2 + e2*const1
        e3 += wf_in[i] - e2 - e3*const2
        e2 = wf_in[i]
        wf_out[i] = e1 - frac*e3
