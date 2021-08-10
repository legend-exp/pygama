import os
import glob
import numpy as np
from numba import guvectorize

    
    
@guvectorize(["void(float32[:], float32[:])",
              "void(float64[:], float64[:])"],
             "(n)->()", nopython=True, cache=True)

def wave_integrator(w_in, q_out):
    """
    Processor to integrate the entire length of a waveform. If any
    input values are nan will return array of nan of w_out.
    Parameters
    ----------
    w_in : array-like
           wf to integrate
    q_out : array-like
            The output value from integrating a wave from the processor
    """
    
    if (np.isnan(w_in).any()):
        return

    q_out[0] = np.sum(w_in[:])
   