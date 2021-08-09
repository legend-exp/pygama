import numpy as np
from numba import guvectorize

@guvectorize(["void(float32[:], uint16, uint16[:], uint16[:])",
              "void(float64[:], uint16, uint16[:], uint16[:])"],
             "(n),()->(),()", nopython=True, cache=True)
def saturation(wf_in, bit_depth, n_lo, n_hi):
    """
    Count the number of samples in the waveform that are
    saturated at the minimum and maximum possible values based
    on the bit depth.
    
    Parameters
    ----------
    wf_in    : array-like
               The input waveform
    bit_depth: int
               The bit depth of the analog-to-digital converter
    n_lo     : int
               The output number of samples at the minimum
    n_hi     : int
               The output number of samples at the maximum
    
    Processing Chain Example
    ------------------------
    "sat_lo, sat_hi": {
        "function": "saturation",
        "module": "pygama.dsp.processors",
        "args": ["waveform", "16", "sat_lo", "sat_hi"],
        "unit": "ADC",
        "prereqs": ["waveform"]
    }
    """
    n_lo[0] = 0
    n_hi[0] = 0
    for sample in wf_in:
        if sample == 0:
            n_lo[0] += 1
        elif sample == np.power(2, bit_depth):
            n_hi[0] += 1
