import numpy as np
from numba import guvectorize
from pygama.dsp.errors import DSPFatal

@guvectorize(["void(float32[:], float32[:], float32[:], float32[:], float32[:])",
              "void(float64[:], float64[:], float64[:], float64[:], float64[:])"],
             "(n)->(),(),(),()", nopython=True, cache=True)

def linear_slope_fit(w_in, mean, stdev, slope, intercept):   
    """
    Calculate mean and standard deviation using Welford's method. In addition, it performs a linear regression and return best fit values for slope and intercept. Note: mean and stdev are computes assuming a flat slope.
    
    Parameters
    ----------
    w_in : array-like
            Input waveform 
    
    a_mean : float
    stdev : float
    slope : float
    intercept : float
    Processing Chain Example
    ------------------------
    
    "bl_mean , bl_std, bl_slope, bl_intercept":{
        "function": "linear_slope_fit",
        "module": "pygama.dsp.processors",
        "args" : ["wf_blsub[0: 1650]", "bl_mean","bl_std", "bl_slope","bl_intercept"],
        "prereqs": ["wf_blsub"],
        "unit": ["ADC","ADC","ADC","ADC"]
        },
    """

    mean[0] = stdev[0] = slope[0] = intercept[0] = np.nan

    if (np.isnan(w_in).any()):
        return

    sum_x = sum_x2 = sum_xy = sum_y = mean[0] = stdev[0] = 0
    isum = len(w_in)

    for i in range(len(w_in)):
        # mean and stdev
        tmp = w_in[i]-mean
        mean += tmp / (i+1)
        stdev += tmp*(w_in[i]-mean)

        # linear regression
        sum_x += i 
        sum_x2 += i*i
        sum_xy += (w_in[i] * i)
        sum_y += w_in[i]

    stdev /= (isum - 1)
    np.sqrt(stdev, stdev)

    slope[0] = (isum * sum_xy - sum_x * sum_y) / (isum * sum_x2 - sum_x * sum_x)
    intercept[0] = (sum_y - sum_x * slope[0])/isum
