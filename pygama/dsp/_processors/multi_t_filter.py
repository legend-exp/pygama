import numpy as np
from numba import guvectorize, njit
from time_point_thresh import time_point_thresh
from get_multi_local_extrema import get_multi_local_extrema


    
@guvectorize(["void(float32[:],float32[:])",
              "void(float64[:],float64[:])",
              "void(int32[:],int32[:])",
              "void(int64[:],int64[:])"],
             "(m),(m)", nopython=True, cache=True)
def remove_duplicates(t_out,vt_min_out):
    """ 
    time_point_thresh has issues with afterpulsing in waveforms that causes  
    an aferpulse peak's tp0 to be sent to 0 or the same index as the tp0 for the first pulse.
    This only happens when the relative minimum between the first pulse and 
    the afterpulse is greater than the threshold. So, we sweep through the array again 
    to ensure there are no duplicate indices. If there are duplicate indicies caused by a
    misidentified tp0 of an afterpulse, we replace its index by that of the corresponding minimum
    found using the njit_get_i_local_extrema function.
    ----------
    t_out : array-like
        the array of indices that we want to remove duplicates from 
    vt_min_out : array-like
        list of indicies of minima that we want to replace duplicates in t_out with
    """

    k=0
    for index1 in range(len(t_out)):
        for index2 in range(len(t_out[index1+1:])):
            if t_out[index1] == t_out[index2+index1+1]: # check if any later indexed values are equal to the earliest instance
                t_out[index2+index1+1] = vt_min_out[k]
        k+=1 # this makes sure that the index of the misidentified afterpulse tp0 is replaced with the correct corresponding minimum
    if not np.isnan(t_out[0]): # makes sure that the first maximum found isn't the start of the waveform
        if int(t_out[0]) == 0: 
            t_out[:] = np.append(t_out[1:],np.nan)


@guvectorize(["void(float32[:], float32[:],float32[:], float32, float32[:])",
              "void(float64[:], float64[:], float64[:], float64, float64[:])",
              "void(int32[:], int32[:], int32[:], int32, int32[:])",
              "void(int64[:], int64[:], int64[:], int64, int64[:])"],
             "(n),(m),(),()->(m)", forceobj=True, cache=True)
def multi_t_filter(w_in, t_arbitrary_in, a_threshold_in, delta_in, t_out):
    """
    Gets list of indices of the start of leading edges of multiple peaks within a waveform.
    Is built to handle afterpulses/delayed cross talk and trains of pulses.
    The multi_t_filter works by calling the vectorized functions 
    "get_i_local_extrema" which returns a list of the maxima and minima in a waveform,
    and then the list of maxima is fed into "time_point_thresh" which returns 
    the final times that waveform is less than a specified threshold. 

    Parameters
    ----------
    w_in : array-like
        the array of data within which the list of tp0s will be found
    t_arbitraty_in : array-like
        A fixed length array of the same dimensions that you need the tp0 list to be
        Used to tell numba what dimension of an array it should expect
    a_threshold_in: scalar 
        threshold to search for using time_point_thresh
    delta_in : scalar
        the absolute level by which data must vary (in one direction) about an
        extremum in order for it to be tagged
    Returns
    -------
    t_out : array-like
        Array of fixed length (padded with nans) that hold the indices of
        the identified initial rise times of peaks in the signal
        
    """
     
    # initialize arrays, padded with the elements we want
    
    t_out[:] = np.nan 
    vt_max_out= np.zeros_like(t_arbitrary_in, dtype=np.float32)
    vt_min_out = np.zeros_like(t_arbitrary_in,dtype=np.float32)
    n_max_out = np.array([0],dtype=np.float32)
    n_min_out = np.array([0],dtype=np.float32)
    flag_out = np.array([0],dtype=np.float32)

    # call get_i_local_extrema to get locations of the maxima and minima within a waveform
    
    get_multi_local_extrema(w_in, delta_in,t_arbitrary_in, vt_max_out,vt_min_out,n_max_out,n_min_out,flag_out)


    # Go through the list of maxima, calling time_point_thresh (the refactored version ignores the nan padding)
    walk_forward = np.array([0],dtype=np.int32)
    time_point_thresh(w_in, a_threshold_in, vt_max_out, walk_forward, t_out)
    # Remove duplicates from the t_out list
    remove_duplicates(t_out,vt_min_out)

