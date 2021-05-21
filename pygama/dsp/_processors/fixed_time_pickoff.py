import numpy as np
from numba import guvectorize
import math
from pygama.dsp.errors import DSPFatal


@guvectorize(["void(float32[:], float32, float32[:])",
              "void(float64[:], float32, float64[:])"],
             "(n),()->()", nopython=True, cache=True)

def fixed_time_pickoff(w_in, t_in, a_out):
    """
    Fixed time pickoff -- gives the waveform value at a fixed time
    Parameters
    ----------
    w_in : array-like
            Input waveform
    t_in : float
            Time point to find value
    a_out : float
            Output value
    Processing Chain Example
    ------------------------
    "trapEftp": {
        "function": "fixed_time_pickoff",
        "module": "pygama.dsp.processors",
        "args": ["wf_trap", "tp_0 + (10*us+2.5*us)", "trapEftp"],
        "unit": "ADC",
        "prereqs": ["wf_trap", "tp_0"]
        },
    """
    
    a_out[0] = np.nan

    if (np.isnan(w_in).any() or np.isnan(t_in)):
        return
    
    if (not int(t_in) in range(len(w_in))):
        return
  
    if (not np.floor(t_in)==t_in):
        raise DSPFatal('Pickoff Time is not an integer')

    a_out[0] = w_in[int(t_in)]
© 2021 GitHub, Inc.
