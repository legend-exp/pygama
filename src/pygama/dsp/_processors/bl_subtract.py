import numpy as np
from numba import guvectorize

from pygama.dsp.errors import DSPFatal


@guvectorize(["void(float32[:], float32, float32[:])",
              "void(float64[:], float64, float64[:])"],
             "(n),()->(n)", nopython=True, cache=True)
def bl_subtract(w_in, a_baseline, w_out):
    """
    Subtract the constant baseline from the entire waveform.

    Parameters
    ----------
    w_in : array-like
        The input waveform
    a_baseline : float
        The baseline value to subtract
    w_out : array-like
        The output waveform with the baseline subtracted

    Examples
    --------
    .. code-block :: json

        "wf_bl": {
            "function": "bl_subtract",
            "module": "pygama.dsp.processors",
            "args": ["waveform", "baseline", "wf_bl"],
            "unit": "ADC",
            "prereqs": ["waveform", "baseline"]
        }
    """
    w_out[:] = np.nan

    if np.isnan(w_in).any() or np.isnan(a_baseline):
        return

    w_out[:] = w_in[:] - a_baseline
