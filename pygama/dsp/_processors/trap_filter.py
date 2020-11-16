import numpy as np
from numba import guvectorize


@guvectorize(["void(float32[:], int32, int32, float32[:])",
              "void(float64[:], int32, int32, float64[:])",
              "void(int32[:], int32, int32, int32[:])",
              "void(int64[:], int32, int32, int64[:])"],
             "(n),(),()->(n)", nopython=True, cache=True)
def trap_filter(wf_in, rise, flat, wf_out):
    """
    Symmetric trapezoidal filter
    """
    wf_out[0] = wf_in[0]
    for i in range(1, rise):
        wf_out[i] = wf_out[i-1] + wf_in[i]
    for i in range(rise, rise+flat):
        wf_out[i] = wf_out[i-1] + wf_in[i] - wf_in[i-rise]
    for i in range(rise+flat, 2*rise+flat):
        wf_out[i] = wf_out[i-1] + wf_in[i] - wf_in[i-rise] - wf_in[i-rise-flat]
    for i in range(2*rise+flat, len(wf_in)):
        wf_out[i] = wf_out[i-1] + wf_in[i] - wf_in[i-rise] - wf_in[i-rise-flat] + wf_in[i-2*rise-flat]

