import numpy as np
from numba import guvectorize

from pygama.dsp.errors import DSPFatal


@guvectorize(["void(float32[:], float32, float32[:], float32[:], float32[:], float32[:], float32[:])",
              "void(float64[:], float64, float64[:], float64[:],float64[:], float64[:], float64[:])"],
             "(n),(),(m),(m),(),(),()", nopython=True, cache=True)
def get_multi_local_extrema(w_in, a_delta_in, vt_max_out, vt_min_out, n_max_out, n_min_out, flag_out):
    """
    Get lists of indices of the local maxima and minima of data
    The "local" extrema are those maxima / minima that have heights / depths of
    at least a_delta_in.
    Converted from MATLAB script at: http://billauer.co.il/peakdet.html

    Parameters
    ----------
    w_in : array-like
        The array of data within which extrema will be found
    a_delta_in : scalar
        The absolute level by which data must vary (in one direction) about an
        extremum in order for it to be tagged
    vt_max_out, vt_min_out : array-like, array-like
        Arrays of fixed length (padded with nans) that hold the indices of
        the identified local maxima and minima
    n_max_out, n_min_out : scalar, scalar
        The number of maxima and minima found in a waveform
    flag_out : scalar
        Returns 1 if there is only one maximum and it is a simple waveform,
        Returns 0 if there are no peaks, or multiple peaks in a waveform
    """

    # prepare output

    vt_max_out[:]= np.nan
    vt_min_out[:] = np.nan
    n_max_out[0] = np.nan
    n_min_out[0] = np.nan
    flag_out[0] = np.nan

    # initialize internal counters

    n_max_counter = 0
    n_min_counter = 0


    # Checks

    if (np.isnan(w_in).any() or np.isnan(a_delta_in)):
        return

    if (not len(vt_max_out)<len(w_in) or not len(vt_min_out)<len(w_in)):
        raise DSPFatal('The length of your return array must be smaller than the length of your waveform')
    if (not a_delta_in >= 0):
        raise DSPFatal('a_delta_in must be positive')

    # now loop over data

    imax, imin = 0, 0
    find_max = True
    for i in range(len(w_in)):

        if w_in[i] > w_in[imax]: imax = i
        if w_in[i] < w_in[imin]: imin = i

        if find_max:
            # if the sample is less than the current max by more than a_delta_in,
            # declare the previous one a maximum, then set this as the new "min"
            if w_in[i] < w_in[imax] - a_delta_in and int(n_max_counter) < int(len(vt_max_out)):
                vt_max_out[int(n_max_counter)]=imax
                n_max_counter += 1
                imin = i
                find_max = False
        else:
            # if the sample is more than the current min by more than a_delta_in,
            # declare the previous one a minimum, then set this as the new "max"
            if w_in[i] > w_in[imin] + a_delta_in and int(n_min_counter) < int(len(vt_min_out)):
                vt_min_out[int(n_min_counter)] = imin
                n_min_counter += 1
                imax = i
                find_max = True

    # set output
    n_max_out[0] = n_max_counter
    n_min_out[0] = n_min_counter

    if n_max_out[0] == 1:
        flag_out[0] = 1
    else:
        flag_out[0] = 0
