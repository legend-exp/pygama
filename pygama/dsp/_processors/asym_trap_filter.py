import numpy as np
from numba import guvectorize

@guvectorize(["void(float32[:], int32, int32, int32, float32[:])",
              "void(float64[:], int32, int32, int32, float64[:])",
              "void(int32[:], int32, int32, int32, int32[:])",
              "void(int64[:], int32, int32, int32, int64[:])"],
             "(n),(),(),()->(n)", forceobj=True, cache=True)
def asymTrapFilter(wf_in, rise, flat, fall, wf_out): #,padAfter=False
    """ Computes an asymmetric trapezoidal filter """
    # wf_out = np.zeros(len(wf_in))
    # for i in range(len(wf_in)-1000):
    #     w1 = rise
    #     w2 = rise+flat
    #     w3 = rise+flat+fall
    #     r1 = np.sum(wf_in[i:w1+i])/(rise)
    #     r2 = np.sum(wf_in[w2+i:w3+i])/(fall)
    #     # if not padAfter:
    #     #     wf_out[i+1000] = r2 - r1
    #     # else:
    #     wf_out[i] = r2 - r1

    wf_out[:] = wf_in[:]
    wf_out[rise:] -= wf_in[:-fall]
    wf_out[rise+flat:] -= wf_in[:-(fall+flat)]
    wf_out[rise+flat+fall:] += wf_in[:-(rise+flat+fall)]
