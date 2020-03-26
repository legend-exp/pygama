import numpy as np
from numba import guvectorize


@guvectorize(["void(float32[:], int32, int32, float32[:])",
              "void(float64[:], int32, int32, float64[:])",
              "void(int32[:], int32, int32, int32[:])",
              "void(int64[:], int32, int32, int64[:])"],
             "(n),(),()->()", nopython=True, cache=True)
def fixed_time_pickoff(wf_in, reference_time, offset_time, value_out):
    """
    Fixed time pickoff-- gives the waveform value corresponding to a fixed time (offset_time) from the reference_time. As an example, reference_time may be the index corresponding to the waveform max
    """

    value_out[0] = wf_in[reference_time + offset_time]
