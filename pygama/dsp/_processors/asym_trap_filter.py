import numpy as np
from numba import guvectorize
import matplotlib.pyplot as plt

@guvectorize(["void(float32[:], int32, int32, int32, float32[:])",
              "void(float64[:], int32, int32, int32, float64[:])",
              "void(int32[:], int32, int32, int32, int32[:])",
              "void(int64[:], int32, int32, int32, int64[:])"],
             "(n),(),(),()->(n)", forceobj=True, cache=True)
def asymTrapFilter(wf_in, rise, flat, fall, wf_out): #,padAfter=False
    """ Computes an asymmetric trapezoidal filter"""

    wf_out[:] = wf_in[:]
    wf_out[rise:] -= wf_in[:-rise]
    wf_out[:] *= float(fall)/rise
    wf_out[rise+flat:] -= wf_in[:-(rise+flat)]
    wf_out[rise+flat+fall:] += wf_in[:-(rise+flat+fall)]
    wf_out[:] /= fall
    np.cumsum(wf_out, out=wf_out, axis=0)
