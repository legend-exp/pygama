import numpy as np
from numba import guvectorize


@guvectorize(["void(float32[:], int32, int32, int32, float32[:])",
              "void(float64[:], int32, int32, int32, float64[:])",
              "void(int32[:], int32, int32, int32, int32[:])",
              "void(int64[:], int32, int32, int32, int64[:])"],
             "(n),(),(),()->()", nopython=True, cache=True)

def trap_pickoff(wf_in, rise, flat, pickoff_time, value_out):
    """
    Gives the value of a trapezoid, normalized by the "rise" (integration) time, at a specific time equal to pickoff_time. Use when the rest of the trapezoid output is extraneous.
    """
    I_1 = 0.
    I_2 = 0.
    start_time = pickoff_time + 1 # the +1 makes slicing prettier
    
    for i in range(start_time-rise, start_time):
        I_1 += wf_in[i]

    for k in range(start_time - 2*rise - flat, start_time - rise - flat):
        I_2 += wf_in[k]

    value_out[0] = (I_1 - I_2)/rise
