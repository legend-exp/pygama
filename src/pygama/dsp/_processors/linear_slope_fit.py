import numpy as np
from numba import guvectorize
from pygama.dsp.errors import DSPFatal

@guvectorize(["void(float32[:], float32[:], float32[:], float32[:], float32[:])",
              "void(float64[:], float64[:], float64[:], float64[:], float64[:])"],
             "(n)->(),(),(),()", nopython=True, cache=True)
def linear_slope_fit(w_in, mean, stdev, slope, intercept):   
    """
    Calculate the mean and standard deviation of the waveform using
    Welford's method as well as the slope an intercept of the waveform
    using linear regression.
    
    Parameters
    ----------
    w_in     : array-like
               The input waveform
    a_mean   : float
               The mean of the waveform
    stdev    : float
               The standard deviation of the waveform
    slope    : float
               The slope of the linear fit
    intercept: float
               The intercept of the linear fit

    Processing Chain Example
    ------------------------
    "bl_mean, bl_std, bl_slope, bl_intercept": {
        "function": "linear_slope_fit",
        "module": "pygama.dsp.processors",
        "args": ["wf_blsub[0:1650]", "bl_mean", "bl_std", "bl_slope", "bl_intercept"],
        "unit": ["ADC", "ADC", "ADC", "ADC"],
        "prereqs": ["wf_blsub"]
    }
    """
    mean     [0] = np.nan
    stdev    [0] = np.nan
    slope    [0] = np.nan
    intercept[0] = np.nan

    if np.isnan(w_in).any():
        return

    sum_x = sum_x2 = sum_xy = sum_y = mean[0] = stdev[0] = 0
    isum = len(w_in)

    for i in range(0, len(w_in), 1):
        # the mean and standard deviation
        temp   = w_in[i] - mean
        mean  += temp / (i + 1)
        stdev += temp * (w_in[i] - mean)

        # linear regression
        sum_x  += i
        sum_x2 += i * i
        sum_xy += (w_in[i] * i)
        sum_y  += w_in[i]

    stdev /= (isum - 1)
    np.sqrt(stdev, stdev)

    slope    [0] = (isum * sum_xy - sum_x * sum_y) / (isum * sum_x2 - sum_x * sum_x)
    intercept[0] = (sum_y - sum_x * slope[0]) / isum
