import numpy as np
from numba import guvectorize
from math import sqrt

@guvectorize(["void(float32[:], float32[:], float32[:])",
              "void(float64[:], float64[:], float64[:])"],
             "(n)->(),()", nopython=True, cache=True)
def mean_stdev(wf, mean, stdev):
    """
    Calculate the mean and standard deviation of a vector using Welford's method
    """
    mean[0]=0
    stdev[0]=0
    for k, sample in enumerate(wf):
        tmp = sample-mean
        mean+=tmp/(k+1)
        stdev+=tmp*(sample-mean)
    stdev/=(len(wf)-1)
    np.sqrt(stdev, stdev)
