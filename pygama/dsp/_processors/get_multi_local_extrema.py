import numpy as np
from numba import guvectorize, njit
from pygama.dsp.errors import DSPFatal



@guvectorize(["void(float32[:], float32, float32[:], float32[:], float32[:], float32[:], float32[:], int32[:])",
              "void(float64[:], float64, float64[:], float64[:], float64[:],float64[:], float64[:], int32[:])",
              "void(int32[:], int32, int32[:], int32[:], int32[:], int32[:], int32[:], int32[:])",
              "void(int64[:], int64, int64[:], int64[:], int64[:], int64[:], int64[:], int32[:])"],
             "(n),(),(m) -> (m),(m),(),(),()", nopython=True, cache=True)
def get_multi_local_extrema(w_in, delta_in, t_arbitrary_in, vt_max_out,vt_min_out,n_max_out,n_min_out,flag_out):
    """
    Get lists of indices of the local maxima and minima of data
    The "local" extrema are those maxima / minima that have heights / depths of
    at least delta.
    Converted from MATLAB script at: http://billauer.co.il/peakdet.html
    Parameters
    ----------
    w_in : array-like
        the array of data within which extrema will be found
    delta_in : scalar
        the absolute level by which data must vary (in one direction) about an
        extremum in order for it to be tagged
    Returns
    -------
    vt_max_out, vt_min_out : array-like, array-like
        Arrays of fixed length (padded with nans) that hold the indices of
        the identified local maxima and minima
    n_max_out, n_min_out: scalar, scalar 
        The number of maxima and minima found in a waveform
    flag_out: scalar
        Returns 0 if there is only one maximum and it is a simple waveform, 
        returns 1 if there are multiple peaks in a waveform
        
    """

    # prepare output 
    #padded with nans or zeros depending on what we want
    vt_max_out[:]= np.nan
    vt_min_out[:] = np.nan

    
    n_max_out[0] = 0
    n_min_out[0] = 0
    flag_out[0] = 0
    

    # sanity checks
    w_in, delta2 = np.asarray(w_in), np.asarray(delta_in)
    if delta2.ndim != 0:
        raise DSPFatal("Input argument delta_in must be a scalar")
        
    if delta_in <= 0:
        raise DSPFatal("Input argument delta_in must be positive")
        

    # now loop over data
    imax, imin = 0, 0
    find_max = True
    for i in range(len(w_in)):

        if w_in[i] > w_in[imax]: imax = i
        if w_in[i] < w_in[imin]: imin = i

        if find_max:
            # if the sample is less than the current max by more than delta,
            # declare the previous one a maximum, then set this as the new "min"
            if w_in[i] < w_in[imax] - delta_in and int(n_max_out[0]) < int(len(vt_max_out)):
                vt_max_out[int(n_max_out[0])]=imax
                n_max_out[0] += 1
                imin = i
                find_max = False
        else:
            # if the sample is more than the current min by more than delta,
            # declare the previous one a minimum, then set this as the new "max"
            if w_in[i] > w_in[imin] + delta_in and int(n_min_out[0]) < int(len(vt_min_out)):
                vt_min_out[int(n_min_out[0])] = imin
                n_min_out[0] += 1
                imax = i
                find_max = True
    if n_max_out[0] != 0:
        flag_out[0] = 1






    

