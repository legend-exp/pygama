import numpy as np
from numba import guvectorize

from pygama.dsp.errors import DSPFatal


@guvectorize(["void(float32[:], float32[:])",
              "void(float64[:], float64[:])"],
             "(n)->(n)", nopython=True, cache=True)
def log_check(w_in, w_log):
    """
    Calculate the logarithm of the waveform if all its values
    are positive; otherwise, return NaN.

    Parameters
    ----------
    w_in : array-like
           The input waveform
    w_log: array-like
           The output waveform with logged values

    Examples
    --------
    .. code-block :: json

        "wf_logged": {
            "function": "log_check",
            "module": "pygama.dsp.processors",
            "args": ["wf_blsub[2100:]", "wf_logged"],
            "unit": "ADC",
            "prereqs": ["wf_blsub"]
        }
    """
    w_log[:] = np.nan

    if np.isnan(w_in).any():
        return

    if np.any(w_in <= 0):
        return

    w_log[:] = np.log(w_in[:])
