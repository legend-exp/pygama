import numpy as np
from numba import guvectorize

from pygama.dsp.errors import DSPFatal


@guvectorize(["void(float32[:], float32, float32[:])",
              "void(float64[:], float64, float64[:])"],
             "(n),()->()", nopython=True, cache=True)
def fixed_time_pickoff(w_in, t_in, a_out):
    """
    Pick off the waveform value at the provided index.  If the
    provided index is out of range, return NaN. For non-integer
    indices, interpolate linearly between nearest samples

    Parameters
    ----------
    w_in : array-like
        The input waveform
    t_in : float
        The waveform index to pick off
    a_out : float
        The output pick-off value

    Examples
    --------
    .. code-block :: json

        "trapEftp": {
            "function": "fixed_time_pickoff",
            "module": "pygama.dsp.processors",
            "args": ["wf_trap", "tp_0+10*us", "trapEftp"],
            "unit": "ADC",
            "prereqs": ["wf_trap", "tp_0"]
        }
    """
    a_out[0] = np.nan

    if np.isnan(w_in).any() or np.isnan(t_in):
        return

    if int(t_in) < 0 or int(t_in) >= len(w_in):
        return

    i_in = int(t_in)
    w = t_in-i
    a_out[0] = (1-w)*w_in[i_in] + w*w_in[i_in+1]
