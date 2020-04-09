import numpy as np
from numba import guvectorize

@guvectorize(["void(float32[:], float32, int32, int32[:])",
              "void(float64[:], float64, int32, int32[:])",
              "void(int32[:], float32, int32, int32[:])",
              "void(int64[:], float64, int32, int32[:])"],
             "(n),(),()->()", nopython=True, cache=True)
def time_point_frac(wf_in, frac, tp_max, tp_out):
    """
    Find the time where the waveform crosses a value that is a fraction of the
    max. Parameters are:
     wf_in: input waveform. Should have baseline of 0!
     frac: fraction of maximum to search for. Should be between 0 and 1.
     tp_max: timepoint of wf maximum. Can be found with numpy.argmax
     tp_out: time that waveform crosses frac * wf_in[tp_max] for the last time,              rounded to an integer index
    """
    threshold = frac*wf_in[tp_max]
    tp_out[0] = tp_max-1
    # Scan for the crossing
    while(wf_in[tp_out[0]] > threshold):
        tp_out[0] -= 1
    # if the previous point is closer to the threshold than the one we landed on
    # use that. This is equivalent to interpolating and then rounding
    if(threshold - wf_in[tp_out[0]] > wf_in[tp_out[0]+1] - threshold):
        tp_out[0] += 1
