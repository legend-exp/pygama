import numpy as np
from numba import guvectorize


@guvectorize(["void(float32[:], float32[:], float32[:], float32[:], float32[:])",
              "void(float64[:], float64[:], float64[:], float64[:], float64[:])"],
             "(n)->(),(),(),()", nopython=True, cache=True)



def min_max(wf, min, max, argmin, argmax):

    min[0] = max[0] = wf[0]
    argmin[0] = argmax[0] = 0

    for i,value in enumerate(wf):
        if value < min[0]:
            min[0] = value
            argmin[0] = i
        elif value > max[0]:
            max[0] = value
            argmax[0] = i
