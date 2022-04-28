import numpy as np
from numba import guvectorize

from pygama.dsp.errors import DSPFatal


@guvectorize(["void(float32[:], float32, float32, float32, float32[:])",
              "void(float64[:], float64, float64, float64, float64[:])"],
             "(n),(),(),()->()", nopython=True, cache=True)
def time_point_thresh(w_in, a_threshold, t_start, walk_forward, t_out):
    """
    Find the index where the waveform value crosses the threshold,
    walking either forward or backward from the starting index.

    Parameters
    ----------
    w_in        : array-like
                  The input waveform
    a_threshold : float
                  The threshold value
    t_start     : int
                  The starting index
    walk_forward: int
                  The backward (0) or forward (1) search direction
    t_out       : float
                  The index where the waveform value crosses the threshold

    Processing Chain Example
    ------------------------
    "tp_0": {
        "function": "time_point_thresh",
        "module": "pygama.dsp.processors",
        "args": ["wf_atrap", "bl_std", "tp_start", 0, "tp_0"],
        "unit": "ns",
        "prereqs": ["wf_atrap", "bl_std", "tp_start"]
    }
    """
    t_out[0] = np.nan

    if np.isnan(w_in).any() or np.isnan(a_threshold) or np.isnan(t_start) or np.isnan(walk_forward):
        return

    if np.floor(t_start) != t_start:
        raise DSPFatal('The starting index must be an integer')

    if np.floor(walk_forward) != walk_forward:
        raise DSPFatal('The search direction must be an integer')

    if int(t_start) < 0 or int(t_start) >= len(w_in):
        raise DSPFatal('The starting index is out of range')

    if int(walk_forward) == 1:
        for i in range(int(t_start), len(w_in) - 1, 1):
            if w_in[i] <= a_threshold < w_in[i+1]:
                t_out[0] = i
                return
    else:
        for i in range(int(t_start), 1, -1):
            if w_in[i-1] < a_threshold <= w_in[i]:
                t_out[0] = i
                return
