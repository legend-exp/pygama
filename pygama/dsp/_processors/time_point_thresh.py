import numpy as np
from numba import guvectorize

@guvectorize(["void(float32[:], float32, int32, int32[:])",
              "void(float64[:], float64, int32, int32[:])",
              "void(int32[:], int32, int32, int32[:])",
              "void(int64[:], int64, int32, int32[:])"],
             "(n),(),()->()", nopython=True, cache=True)
def time_point_thresh(wf_in, threshold, tp_max, tp_out):
    """
    Find the last timepoint before tp_max that wf_in crosses a threshold
     wf_in: input waveform
     threshold: threshold to search for
     tp_out: final time that waveform is less than threshold
    """
    for i in range(tp_max, 0, -1):
        if(wf_in[i]>threshold and wf_in[i-1]<threshold):
            tp_out[0] = i
            return
    tp_out[0] = 0
    return
