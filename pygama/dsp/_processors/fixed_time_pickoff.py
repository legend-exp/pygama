import numpy as np
from numba import guvectorize


@guvectorize(["void(float32[:], int32, float32[:])",
              "void(float64[:], int32, float64[:])",
              "void(int32[:], int32, int32[:])",
              "void(int64[:], int32, int64[:])"],
             "(n),()->()", nopython=True, cache=True)
def fixed_time_pickoff(wf_in, time, value_out):
    """
    Fixed time pickoff-- gives the waveform value at a fixed time
    """

    value_out[0] = wf_in[time]
