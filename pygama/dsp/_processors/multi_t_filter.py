import numpy as np
from numba import guvectorize, njit

"""
This block of code creates a filter that returns a list of tp0's of multiple peaks within a waveform. 
It is built to handle waveforms with afterpulsing/cross-talk as well trains of pulses. 
The vectorized multi_triggering_filter works by calling the njit'ed functions 
(because guvectorized functions cannot call other guvectorized functions)
"njit_get_i_local_extrema" which returns a list of the maxima and minima in a waveform,
and then the maxima are individually fed into "njit_time_point_thresh" which returns 
the final times that waveform is less than a specified threshold. 
"""

@njit
def njit_get_i_local_extrema(data, delta,max_list,min_list,n_max,n_min,flag):
    """
    Get lists of indices of the local maxima and minima of data
    The "local" extrema are those maxima / minima that have heights / depths of
    at least delta.
    Converted from MATLAB script at: http://billauer.co.il/peakdet.html
    Parameters
    ----------
    data : array-like
        the array of data within which extrema will be found
    delta : scalar
        the absolute level by which data must vary (in one direction) about an
        extremum in order for it to be tagged
    Returns
    -------
    max_list, min_list : array-like, array-like
        Arrays of fixed length (padded with nans) that hold the indices of
        the identified local maxima and minima
    n_max, n_min: scalar, scalar 
        The number of maxima and minima found in a waveform
    flag: scalar
        Returns 0 if there is only one maximum and it is a simple waveform, 
        returns 1 if there are multiple peaks in a waveform
        
    """

    # prepare output 
    # padded with nans or zeros depending on what we want
    max_list[:]= np.nan
    min_list[:] = np.nan
    n_max[0] = 0
    n_min[0] = 0
    flag[0] = 0
    

    # sanity checks
    data, delta2 = np.asarray(data), np.asarray(delta)
    if delta2.ndim != 0:
        raise Exception("Input argument delta must be a scalar")
        return np.array([]), np.array([])
    if delta <= 0:
        raise Exception("Input argument delta must be positive")
        return np.array([]), np.array([])

    # now loop over data
    imax, imin = 0, 0
    find_max = True
    for i in range(len(data)):

        if data[i] > data[imax]: imax = i
        if data[i] < data[imin]: imin = i

        if find_max:
            # if the sample is less than the current max by more than delta,
            # declare the previous one a maximum, then set this as the new "min"
            if data[i] < data[imax] - delta and int(n_max[0]) < int(len(max_list)):
                max_list[int(n_max[0])]=imax
                n_max[0] += 1
                imin = i
                find_max = False
        else:
            # if the sample is more than the current min by more than delta,
            # declare the previous one a minimum, then set this as the new "max"
            if data[i] > data[imin] + delta and int(n_min[0]) < int(len(min_list)):
                min_list[int(n_min[0])] = imin
                n_min[0] += 1
                imax = i
                find_max = True
    if n_max[0] != 0:
        flag[0] = 1



@njit
def njit_time_point_thresh(wf_in, threshold, tp_max, tp_out):
    """
    Find the last timepoint before tp_max that wf_in crosses a threshold
     wf_in: input waveform
     threshold: threshold to search for
     tp_max: time of a maximum of a waveform that the search starts at
     tp_out: final time that waveform is less than threshold
    """
    for i in range(tp_max, 0, -1):
        if(wf_in[i]>threshold and wf_in[i-1]<threshold):
            tp_out = i
            return tp_out
    tp_out = 0
    return tp_out
    


@guvectorize(["void(float32[:], float32[:],float32[:], float32, float32[:])",
              "void(float64[:], float64[:], float64[:], float64, float64[:])",
              "void(int32[:], int32[:], int32[:], int32, int32[:])",
              "void(int64[:], int64[:], int64[:], int64, int64[:])"],
             "(n),(m),(),()->(m)", nopython=True, cache=True)
def multi_triggering_filter(wf_in, arbitary_list, threshold, delta, tp0_list):
    """
    Gets list of indices of the start of leading edges of multiple peaks within a waveform.
    Is built to handle afterpulses/delayed cross talk and trains of pulses.
    Parameters
    ----------
    wf_in : array-like
        the array of data within which the list of tp0s will be found
    arbitraty_list : array-like
        A fixed length array of the same dimensions that you need the tp0 list to be
        Used to tell numba what dimension of an array it should expect
    threshold: scalar 
        threshold to search for using time_point_thresh
    delta : scalar
        the absolute level by which data must vary (in one direction) about an
        extremum in order for it to be tagged
    Returns
    -------
    tp0_list : array-like
        Array of fixed length (padded with nans) that hold the indices of
        the identified initial rise times of peaks in the signal
        
    """
     
    # initialize arrays, padded with the elements we want
    
    tp0_list[:] = np.nan 
    max_list= np.zeros_like(tp0_list)
    min_list = np.zeros_like(tp0_list)
    n_max = np.array([0],dtype=np.int32)
    n_min = np.array([0],dtype=np.int32)
    flag = np.array([0],dtype=np.int32)

    # call njit_get_i_local_extrema to get locations of the maxima and minima within a waveform
    
    njit_get_i_local_extrema(wf_in, delta,max_list,min_list,n_max,n_min,flag)
    
    # Go through the list of maxima, calling njit_time_point_thresh on each maximum (ignoring the nan padding)
    for i in range(int(n_max[0])):
        if not np.isnan(max_list[i]): 
            tp0_list[i] = njit_time_point_thresh(wf_in,float(threshold[0]),int(max_list[i]),tp0_list[i])
            
    """ 
    time_point_thresh has issues with afterpulsing in waveforms that causes  
    an aferpulse peak's tp0 to be sent to 0 or the same index as the tp0 for the first pulse.
    This only happens when the relative minimum between the first pulse and 
    the afterpulse is greater than the threshold. So, we sweep through the array again 
    to ensure there are no duplicate indices. If there are duplicate indicies caused by a
    misidentified tp0 of an afterpulse, we replace its index by that of the corresponding minimum
    found using the njit_get_i_local_extrema function.
    """

    k=0
    for index, i in enumerate(tp0_list):
        for index2, j in enumerate(tp0_list[index+1:]):
            if i == j:
                tp0_list[index2+index+1] = min_list[k]
        k+=1 # this makes sure that the index of the misidentified afterpulse tp0 is replaced with the correct corresponding minimum
            