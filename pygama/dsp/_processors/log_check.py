import numpy as np
from numba import guvectorize

@guvectorize(["void(float32[:], float32[:])",
              "void(float64[:], float64[:])"],
             "(n)->(n)", nopython=True, cache=True)


def log_check(wf, log_wf):
    '''
    This processor will take in a slice of the baseline subtracted waveform and returns the log of this 
    slice if all are positive and an array of nan if any value is negative
    '''

    if np.any(wf<0) == True:
        log_wf[0:len(wf)] = np.nan
    else:
        log_wf[0:] = np.log(wf[0:])
