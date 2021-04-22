import numpy as np
from numba import guvectorize

@guvectorize(["void(float32[:], float64, float32[:])",
              "void(float64[:], float64, float64[:])"],
             "(n),()->()", nopython=True, cache=True)
def fixed_time_pickoff(wf_in, time, value_out):
    """
    Return the waveform value at the requested time. Interpolate between
    samples for floating point times.
    """
    if not (time >= 0 and time <= len(wf_in-1)):
        value_out[0] = np.nan
    
    t = int(time)
    frac = time-t
    if frac==0:
        value_out[0] = wf_in[t]
    else:
        value_out[0] = wf_in[t]*(1-frac) + wf_in[t+1]*frac
