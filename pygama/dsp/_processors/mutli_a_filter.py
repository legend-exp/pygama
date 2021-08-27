import numpy as np
from numba import guvectorize
from pygama.dsp.errors import DSPFatal
from fixed_time_pickoff import fixed_time_pickoff
from get_multi_local_extrema import get_multi_local_extrema

@guvectorize(["void(float32[:], float32[:], float32, float32[:])",
              "void(float64[:], float64[:], float64, float64[:])"],
             "(n),(m),()->(m)", forceobj=True, cache=True)
def multi_a_filter(w_in, t_arbitrary_in, a_delta_in, va_max_out): 
    """
    Finds the maximums in a waveform and returns the amplitude of the wave at those points
    
    Parameters
    ----------
    w_in : array-like
        The array of data within which amplitudes of extrema will be found 
    t_arbitrary_in : array-like
        An array of fixed length that tells numba to return the list of amplitudes of the same length
    a_delta_in : scalar
        The absolute level by which data must vary (in one direction) about an
        extremum in order for it to be tagged
        
    Returns
    -------
    va_max_out: array-like
        An array of the amplitudes of the maximums of the waveform
    """
    
    # Initialize output parameters 
    
    va_max_out[:] = np.nan
    
    # Check inputs 
    
    if (np.isnan(w_in).any() or np.isnan(a_delta_in)):
        return

    if not a_delta_in >= 0:
        raise DSPFatal('Error Message: a_delta_in must be positive')
        
    if len(t_arbitrary_in) != len(va_max_out): 
        raise DSPFatal('Error Message: Output arrays must be the same length as the arbitary input array')
        
    if (not len(t_arbitrary_in)<len(w_in)):
        raise DSPFatal('The length of your return array must be smaller than the length of your waveform')
    
    # Use get_multi_local_extrema to find vt_max_out for a waveform
    
    vt_max_out = np.full_like(t_arbitrary_in, np.nan, dtype=np.float32)
    vt_min_out = np.full_like(t_arbitrary_in, np.nan, dtype=np.float32)
    n_max_out = np.array([np.nan], dtype=np.float32)
    n_min_out = np.array([np.nan], dtype=np.float32)
    flag_out = np.array([np.nan], dtype=np.float32)
    get_multi_local_extrema(w_in, a_delta_in, t_arbitrary_in, vt_max_out, vt_min_out, n_max_out, n_min_out, flag_out)
    
    # Feed vt_max_out into fixed_time_pickoff to get va_max_out
    
    fixed_time_pickoff(w_in, vt_max_out, va_max_out)