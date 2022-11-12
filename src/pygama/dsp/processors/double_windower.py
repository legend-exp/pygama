from __future__ import annotations

import numpy as np
from numba import guvectorize

from pygama.dsp.errors import DSPFatal


@guvectorize(
    [
        "void(float32[:], float32, float32, float32[:])",
        "void(float64[:], float64, float64, float64[:])",
    ],
    "(n),(), (), (m)",
    nopython=True,
    cache=True,
)
def double_windower(
    w_in: np.ndarray, t0_in: int, tf_in: int, w_out: np.ndarray
) -> None:
    """
    Return a shorter sample of the waveform, starting and stopping at the
    specified indices.  Note that the length of the output waveform
    is determined by the length of "w_out" rather than an input
    parameter. If the length of "w_out" plus "t0_in" plus "tf_in" is
    different from the length of "w_in", a DSPFatal is raised.


    Parameters
    ----------
    w_in
        The input waveform
    t0_in
        The starting index of the window
    tf_in
        The index to stop the windowing, measured from the end of the w_in
    w_out
        The windowed waveform

    Notes
    -----
    For normal dsp routines, it is better to use windower twice; this way, the output array is padded with numpy nans
    Use this if you don't want an output array that contains nan values, such as in circumstances where memory usage is critical.
    """
    w_out[:] = np.nan

    if np.isnan(w_in).any() or np.isnan(t0_in) or np.isnan(tf_in):
        return

    if len(w_out) >= len(w_in):
        raise DSPFatal("The windowed waveform must be smaller than the input waveform")

    if len(w_out) + t0_in + tf_in != len(w_in):
        raise DSPFatal(
            "The windowed waveform length must be equal the input waveform length minus t0_in minus tf_in"
        )

    if t0_in < 0 or t0_in > len(w_in):
        raise DSPFatal("The start of the window must be inside the waveform")
    if tf_in < 0 or tf_in > len(w_in):
        raise DSPFatal(
            "The end of the window must be inside the waveform, and tf_in must be positive"
        )
    if t0_in > len(w_in) - tf_in:
        raise DSPFatal("t0_in must occur before tf_in")

    w_out[:] = w_in[int(t0_in) : -1 * int(tf_in)]
