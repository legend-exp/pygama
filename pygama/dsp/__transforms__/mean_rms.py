import numpy as np
from numba import guvectorize


@guvectorize(["void(float32[:], float32[:], float32[:])",
              "void(float64[:], float64[:], float64[:])"],
             "(n)->(),()", nopython=True, cache=True)
def mean_rms(wf, mean, var):
    """
    Calculate the mean and standard deviation of a vector
    """
    mean[0]=0
    var[0]=0
    for sample in wf:
        mean+=sample
        var+=sample*sample
    mean/=len(wf)
    var/=len(wf)
    var=np.sqrt(var-mean*mean)
