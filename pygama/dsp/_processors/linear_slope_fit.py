import numpy as np
from numba import guvectorize

@guvectorize(["void(float32[:], float32[:], float32[:], float32[:], float32[:])",
              "void(float64[:], float64[:], float64[:], float64[:], float64[:])"],
             "(n)->(),(),(),()", nopython=True, cache=True)



def linear_slope_fit(wf, mean_y, sigma_y, slope, intercept):   

    """Finds slope params, mean and stdev of wavefunction slice input"""

    sum_x = sum_x2 = sum_xy = sum_y = mean_y[0] = sigma_y[0] = 0
    isum = len(wf)

    for i,value in enumerate(wf):
        sum_x += i 
        sum_x2 += i**2
        sum_xy += (value * i)
        sum_y += value
        mean_y += (value-mean_y) / (i+1)
        sigma_y += (value-mean_y)**2


    sigma_y /= (isum + 1)
    np.sqrt(sigma_y, sigma_y)


    slope[0] = (isum * sum_xy - sum_x * sum_y) / (isum * sum_x2 - sum_x * sum_x)
    intercept[0] = (sum_y - sum_x * slope[0])/isum
