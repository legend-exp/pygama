import numpy as np
from numba import guvectorize
from pygama.dsp.errors import DSPFatal
from pygama.dsp._processors.fixed_time_pickoff import fixed_time_pickoff

@guvectorize(["void(float32[:], float32[:], float32[:])",
              "void(float64[:], float64[:], float64[:])"],
             "(n),(m)->(m)", forceobj=True, cache=True)
def multi_a_filter(w_in, vt_maxs_in, va_max_out): 
    """
    Finds the maximums in a waveform and returns the amplitude of the wave at those points
    
    Parameters
    ----------
    w_in : array-like
        The array of data within which amplitudes of extrema will be found 
    vt_maxs_in : array-like
        The array of max positions for each wf
    Returns
    -------
    va_max_out: array-like
        An array of the amplitudes of the maximums of the waveform
    """
    
    # Initialize output parameters 
    
    va_max_out[:] = np.nan
    
    # Check inputs 
    
    if (np.isnan(w_in).any()):
        return
        
    if (not len(vt_maxs_in)<len(w_in)):
        raise DSPFatal('The length of your return array must be smaller than the length of your waveform')
    
    fixed_time_pickoff(w_in, vt_maxs_in, va_max_out)