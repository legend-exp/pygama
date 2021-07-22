import numpy as np
from numba import guvectorize
from pygama.dsp.errors import DSPFatal

@guvectorize(["void(float32[:], uint16, float32[:])",
              "void(float64[:], uint16, float64[:])"],
             "(n),()->(n)", nopython=True, cache=True)

def bl_subtract(w_in, a_baseline, w_out):
    """
    Processor to subtract the fpga baseline from all waveform values. If any
    input values are nan will return array of nan of w_out.
    Parameters
    ----------
    w_in : array-like
           wf to baseline subtract
    a_in : uint
           The baseline values to subtract, these are stored as uints in the raw files
    w_out : array-like
            The output waveform from the processor
    Processing Chain Example
    ------------------------
    "wf_blsub":{
        "function": "bl_subtract",
        "module": "pygama.dsp.processors",
        "args": ["waveform", "baseline", "wf_blsub"],
        "prereqs": ["waveform", "baseline"],
        "unit": "ADC"
        },
    """

    w_out[:] = np.nan

    if (np.isnan(w_in).any() or np.isnan(a_baseline)):
        return

    if (not a_baseline >= 0):
        raise DSPFatal('a_baseline is out of range')

    w_out[:] = w_in[:] - a_baseline
