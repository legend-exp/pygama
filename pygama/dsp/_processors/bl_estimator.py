import numpy as np
from numba import guvectorize


@guvectorize(["void(float32[:], float32[:])",
              "void(float64[:], float64[:])"],
             "(n)->(n)", nopython=True, cache=True)

def bl_estimator(w_in, w_out):
    """
    Processor to calculate and subtract a baseline for all waveform values. If any
    input values are nan will return array of nan of w_out. The baseline is caclulated
    by taking the average of the first and last 250 samples.
    Parameters
    ----------
    w_in : array-like
           wf to baseline subtract
    w_out : array-like
            The output waveform from the processor
    """
    
    w_out[:] = np.nan
    if (np.isnan(w_in).any()):
        return
    if (len(w_in)<250):
        raise Exception('window of averaging is out of range')
        
    w_out[:] = w_in[:] - (w_in[0:250].mean()+w_in[-250:].mean())/2
  