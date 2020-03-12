import numpy as np
from numba import guvectorize
import matplotlib.pyplot as plt


@guvectorize(["void(float32[:], int32, int32, float32[:])",
              "void(float64[:], int32, int32, float64[:])",
              "void(int32[:], int32, int32, int32[:])",
              "void(int64[:], int32, int32, int64[:])"],
             "(n),(),()->(n)", forceobj=True, cache=True)
def trap_filter(wf_in, rise, flat, wf_out):
    """
    Symmetric trapezoidal filter
    """
    wf_out[:] = wf_in[:]
    wf_out[rise:] -= wf_in[:-rise]
    wf_out[rise+flat:] -= wf_in[:-(rise+flat)]
    wf_out[2*rise+flat:] += wf_in[:-(2*rise+flat)]
    wf_out = np.cumsum(wf_out, out=wf_out, axis=0)
    x = np.arange(0,3000)
    plt.plot(x, wf_in)
    plt.plot(x, wf_out)
    plt.show()
    exit()


    #Plot every line
