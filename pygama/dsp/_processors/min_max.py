import numpy as np
from numba import guvectorize


@guvectorize(["void(float32[:], int32[:], int32[:],float32[:], float32[:])",
              "void(float64[:], int64[:], int64[:],float64[:], float64[:] )"],
             "(n)->(),(),(),()", nopython=True, cache=True)



def min_max(wf, argmin, argmax, wf_min, wf_max, ):
    '''
    Finds the min, max and their time position for a waveform
    '''

    
    wf_min[0] = wf[0]
    wf_max[0] = wf[0]
    argmin[0] = 0
    argmax[0] = 0

    for i,value in enumerate(wf):
        if value < wf_min[0]:
            wf_min[0] = value
            argmin[0] = i
        elif value > wf_max[0]:
            wf_max[0] = value
            argmax[0] = i
