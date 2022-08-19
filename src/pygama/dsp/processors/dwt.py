from __future__ import annotations

from typing import Callable

import numpy as np
from numba import guvectorize
from pywt import downcoef

from pygama.dsp.errors import DSPFatal


def dwt(wave_type: str, level: int) -> Callable:
    """
    Apply a discrete wavelet transform to the waveform and return only
    the approximate coefficients.

    Note
    ----
    This processor is composed of a factory function that is called using the
    ``init_args`` argument. The input and output waveforms are passed using
    ``args``. The output waveform dimension must be specified.


    Parameters
    ----------
    wave_type
       The wavelet type for discrete convolution ('haar', 'db1', ...)
    level
       The level of decompositions to be performed (1, 2, ...)


    JSON Configuration Example
    --------------------------
    .. code-block :: json

        "dwt_haar":{
            "function": "dwt",
            "module": "pygama.dsp.processors",
            "args": ["wf_blsub", "dwt_haar(250)"],
            "unit": "ADC",
            "prereqs": ["wf_blsub"],
            "init_args": ["'haar'", "3",]
        }
    """

    if level <= 0:
        raise DSPFatal("The level must be a positive integer")

    @guvectorize(
        ["void(float32[:], float32[:])", "void(float64[:], float64[:])"],
        "(n),(m)",
        forceobj=True,
    )
    def dwt_out(wf_in: np.ndarray, wf_out: np.ndarray) -> None:
        """
        Parameters
        ----------
        wf_in
           The input waveform
        wf_out
           The approximate coefficients. The dimension of this array can be calculated
           by out_dim = wf_length/(filter_length^level), where filter_length can be
           obtained via pywt.Wavelet(wave_type).dec_len
        """
        wf_out[:] = np.nan

        if np.isnan(wf_in).any():
            return

        wf_out[:] = downcoef("a", wf_in, wave_type, level=level)

    return dwt_out
