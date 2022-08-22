from __future__ import annotations

import numpy as np
from numba import guvectorize

from pygama.dsp.errors import DSPFatal
from pygama.dsp.utils import numba_defaults_kwargs as nb_kwargs

from .time_point_thresh import time_point_thresh


@guvectorize(
    [
        "void(float32[:],float32[:],float32[:])",
        "void(float64[:],float64[:],float64[:])",
    ],
    "(n),(n) -> (n)",
    **nb_kwargs,
    forceobj=True,
)
def remove_duplicates(
    t_in: np.ndarray, vt_min_in: np.ndarray, t_out: np.ndarray
) -> None:
    """Helper function to remove duplicate peak positions.

    :func:`.time_point_thresh` has issues with afterpulsing in waveforms that
    causes an aferpulse peak position to be sent to zero or the same index as
    the first pulse. This only happens when the relative minimum between the
    first pulse and the afterpulse is greater than the threshold. So, we sweep
    through the array again to ensure there are no duplicate indices. If there
    are duplicate indices caused by a misidentified position of an afterpulse,
    we replace its index by that of the corresponding minimum found using
    :func:`.get_multi_local_extrema`. It also checks to make sure that the
    maximum of a waveform isn't right at index 0.

    Parameters
    ----------
    t_in
        the array of indices that we want to remove duplicates from.
    vt_min_in
        list of indices of minima that we want to replace duplicates in `t_out`
        with.
    t_out
        the array we want to return that will have no duplicate indices in it.

    See Also
    --------
    .multi_t_filter
    """
    # initialize arrays
    t_out[:] = np.nan

    # checks
    if (
        np.isnan(t_in).all() and np.isnan(vt_min_in).all()
    ):  # we pad these with NaNs, so only return if there is nothing to analyze
        return

    # check if any later indexed values are equal to the earliest instance
    k = 0
    for index1 in range(len(t_in)):
        for index2 in range(len(t_in[index1 + 1 :])):
            if t_in[index1] == t_in[index2 + index1 + 1]:
                t_out[index2 + index1 + 1] = vt_min_in[k]
        k += 1  # this makes sure that the index of the misidentified afterpulse tp0 is replaced with the correct corresponding minimum

    # Fill up the output with the rest of the values from the input that weren't repeats
    for index in range(len(t_in)):
        if np.isnan(t_out[index]) and not np.isnan(t_in[index]):
            t_out[index] = t_in[index]

    # makes sure that the first maximum found isn't the start of the waveform
    if not np.isnan(t_out[0]):
        if int(t_out[0]) == 0:
            t_out[:] = np.append(t_out[1:], np.nan)


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:], float64[:], float64[:])",
    ],
    "(n),(),(m),(m),(m)",
    **nb_kwargs,
    forceobj=True,
)
def multi_t_filter(
    w_in: np.ndarray,
    a_threshold_in: float,
    vt_max_in: np.ndarray,
    vt_min_in: np.ndarray,
    t_out: np.ndarray,
) -> None:
    """Gets list of indices of the start of leading edges of multiple peaks
    within a waveform.

    Is built to handle afterpulses/delayed cross-talk and trains of pulses.
    Works by calling the vectorized functions :func:`.get_multi_local_extrema`
    which returns a list of the maxima and minima in a waveform, and then the
    list of maxima is fed into :func:`.time_point_thresh` which returns the
    final times that waveform is less than a specified threshold.

    Parameters
    ----------
    w_in
        the array of data within which the list of leading edge times will be
        found.
    a_threshold_in
        threshold to search for using :func:`.time_point_thresh`.
    vt_max_in
        the array of maximum positions for each waveform.
    vt_min_in
        the array of minimum positions for each waveform.
    t_out
        output array of fixed length (padded with :any:`numpy.nan`) that hold
        the indices of the identified initial rise times of peaks in the
        signal.

    See Also
    --------
    ~.time_point_thresh.time_point_thresh
    """

    # initialize arrays, padded with the elements we want
    t_out[:] = np.nan

    # checks
    if np.isnan(w_in).any() or np.isnan(a_threshold_in):
        return
    if np.isnan(vt_max_in).all() and np.isnan(vt_min_in).all():
        return
    if not len(t_out) <= len(w_in):
        raise DSPFatal(
            "The length of your return array must be smaller than the length of your waveform"
        )

    # Initialize an intermediate array to hold the tp0 values before we remove duplicates from it
    intermediate_t_out = np.full_like(t_out, np.nan, dtype=np.float32)

    # Go through the list of maxima, calling time_point_thresh (the refactored version ignores the nan padding)
    time_point_thresh(w_in, a_threshold_in, vt_max_in, 0, intermediate_t_out)

    # Remove duplicates from the t_out list
    remove_duplicates(intermediate_t_out, vt_min_in, t_out)
