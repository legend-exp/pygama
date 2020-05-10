import numpy as np
from numba import guvectorize
import math
from math import pow

def cusp_filter(wsize, sigma, flat, decay):
    """
    CUSP filter
    """
    lenght = int(wsize - 100)
    lt = int((lenght-flat)/2)
    cusp = np.zeros(lenght)
    ind = 0
    while ind < lenght:
        if ind < lt:
            cusp[ind] = float(math.sinh(ind/sigma)/math.sinh(lt/sigma))
        elif ind < lt+flat+1:
            cusp[ind] = 1
        else:
            cusp[ind] = float(math.sinh((lenght-ind)/sigma)/math.sinh(lt/sigma))
        ind += 1
    den = [1, -np.exp(-1/decay)]
    cuspd = np.convolve(cusp, den, 'same')
    
    @guvectorize(["void(float32[:], float32[:])",
                  "void(float64[:], float64[:])",
                  "void(int32[:], int32[:])",
                  "void(int64[:], int64[:])"],
                 "(n),(m)", forceobj=True)
    def cusp_out(wf_in,wf_out):
        wf_out[:] = np.convolve(wf_in, cuspd, 'valid')
    return cusp_out
