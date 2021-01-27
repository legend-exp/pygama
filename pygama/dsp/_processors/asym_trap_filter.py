import numpy as np
from numba import guvectorize

@guvectorize(["void(float32[:], int32, int32, int32, float32[:])",
              "void(float64[:], int32, int32, int32, float64[:])",
              "void(int32[:], int32, int32, int32, int32[:])",
              "void(int64[:], int32, int32, int32, int64[:])"],
             "(n),(),(),()->(n)", nopython=True, cache=True)
def asymTrapFilter(wf_in, rise, flat, fall, wf_out):
    """ Computes an asymmetric trapezoidal filter"""
    wf_out[0] = wf_in[0]/float(rise)
    for i in range(1, rise):
        wf_out[i] = wf_out[i-1] + (wf_in[i])/float(rise)
    for i in range(rise, rise+flat):
        wf_out[i] = wf_out[i-1] + (wf_in[i] - wf_in[i-rise])/float(rise)
    for i in range(rise+flat, rise+flat+fall):
        wf_out[i] = wf_out[i-1] + (wf_in[i] - wf_in[i-rise])/float(rise) - wf_in[i-rise-flat]/float(fall)
    for i in range(rise+flat+fall, len(wf_in)):
        wf_out[i] = wf_out[i-1] + (wf_in[i] - wf_in[i-rise])/float(rise) - (wf_in[i-rise-flat] - wf_in[i-rise-flat-fall])/float(fall)
