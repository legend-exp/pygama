import numpy as np
from numba import guvectorize
import math
from math import pow

@guvectorize(["void(float32[:], int32, int32, int32, float32[:])",
              "void(float64[:], int32, int32, int32, float64[:])",
              "void(int32[:], int32, int32, int32, int32[:])",
              "void(int64[:], int32, int32, int32, int64[:])"],
             "(n),(),(),(),(m)", forceobj=True, cache=True)
def zac_filter(wf_in, sigma, flat, decay, wf_out):
    """
    ZAC filter
    """
    nbin = wf_in.shape[0]
    lenght = nbin - 100
    lt = int((lenght-flat)/2)
    # calculate cusp filter and negative parables
    cusp = np.zeros(lenght)
    par = np.zeros(lenght)
    ind = 0
    while ind < lenght:
        if ind < lt:
            cusp[ind] = float(math.sinh(ind/sigma)/math.sinh(lt/sigma))
            par[ind] = pow(ind-lt/2,2)-pow(lt/2,2)
        elif ind < lt+flat+1:
            cusp[ind] = 1
        else:
            cusp[ind] = float(math.sinh((lenght-ind)/sigma)/math.sinh(lt/sigma))
            par[ind] = pow(lenght-ind-lt/2,2)-pow(lt/2,2)
        ind += 1
    # calculate area of cusp and parables
    areapar, areacusp = 0, 0
    for i in range(lenght):
        areapar += par[i]
        areacusp += cusp[i]
    #normalize parables area
    par = -par/areapar*areacusp
    #create zac filter
    zac = cusp + par
    #deconvolve zac filter
    den = [1, -np.exp(-1/decay)]
    zacd = np.convolve(zac, den, 'same')
    #output
    wf_out[:] = np.convolve(wf_in, zacd, 'valid')
