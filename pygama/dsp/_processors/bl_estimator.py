import numpy as np
from numba import guvectorize
from pygama.dsp.errors import DSPFatal


@guvectorize(["void(float32[:],int32[:], float32[:])",
              "void(float64[:], int64[:], float64[:])"],
             "(n),()->(n)", nopython=True, cache=True)

def bl_estimator(w_in, window, w_out):
    """
    Processor to calculate and subtract a baseline for all waveform values. If any
    input values are nan will return array of nan of w_out. The baseline is caclulated
    by taking the average of the first and last 250 samples.
    Parameters
    ----------
    w_in : array-like
           wf to baseline subtract
    window : scalar
           a number of samples from the start of a wf to average over, as well the number of samples before the end of the waveform to average over
    w_out : array-like
            The output waveform from the processor
    """
    
    w_out[:] = np.nan
    if (np.isnan(w_in).any()):
        return
    if (len(w_in)<int(window[0])):
        raise DSPFatal('window of averaging is out of range')
        
    w_out[:] = w_in[:] - (w_in[0:int(window[0])].mean()+w_in[-int(window[0]):].mean())/2
  