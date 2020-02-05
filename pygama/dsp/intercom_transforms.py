import numpy as np
from numba import guvectorize

@guvectorize(["void(float32[:], float32[:], float32[:])",
              "void(float64[:], float64[:], float64[:])"],
             "(n)->(),()", nopython=True, cache=True)
def mean_sigma(wf, mean, var):
    """
    Calculate the mean and variance of a vector
    """
    mean[0]=0
    var[0]=0
    for sample in wf:
        mean+=sample
        var+=sample*sample
    mean/=len(wf)
    var/=len(wf)
    var=np.sqrt(var-mean*mean)


@guvectorize(["void(float32[:], float32, float32[:])",
              "void(float64[:], float64, float64[:])"],
             "(n),()->(n)", forceobj=True, cache=True)
def polezero(wf_in, tau, wf_out):
    """
    Pole-zero correction using time constant tau
    """
    np.copy_to(wf_out, wf_in)
    np.multiply(wf_out, const, wf_out)
    np.subtract(wf_in[1::-1], wf_out[:-1:-1], wf_out[1::-1])
    np.copy_to(wf_out[0], wf_in[0])
    np.cumsum(wf_out, out=wf_out, axis=1)

@guvectorize(["void(float32[:], int32, int32, float32[:])",
              "void(float64[:], int32, int32, float64[:])",
              "void(int32[:], int32, int32, int32[:])",
              "void(int64[:], int32, int32, int64[:])"],
             "(n),(),()->(n)", forceobj=True, cache=True)
def trapfilter(wf_in, rise, flat, wf_out):
    """
    Symmetric trapezoidal filter
    """
    wf_out[:] = wf_in[:]
    wf_out[rise:] -= wf_in[:-rise]
    wf_out[rise+flat:] -= wf_in[:-(rise+flat)]
    wf_out[2*rise+flat:] += wf_in[:-(2*rise+flat)]
    np.cumsum(wf_out, out=wf_out, axis=0)
