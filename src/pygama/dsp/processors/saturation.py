from __future__ import annotations

import numpy as np
from numba import guvectorize

from pygama.dsp.errors import DSPFatal
from pygama.dsp.utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    [
        "void(float32[:], float32, float32[:], float32[:])",
        "void(float64[:], float64, float64[:], float64[:])",
    ],
    "(n),()->(),()",
    **nb_kwargs,
)
def saturation(
    w_in: np.ndarray, bit_depth_in: int, n_lo_out: int, n_hi_out: int
) -> None:
    """Count the number of samples in the waveform that are saturated at the
    minimum and maximum possible values based on the bit depth.

    Parameters
    ----------
    w_in
        the input waveform
    bit_depth_in
        the bit depth of the analog-to-digital converter
    n_lo_out
        the output number of samples at the minimum
    n_hi_out
        the output number of samples at the maximum

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "sat_lo, sat_hi": {
            "function": "saturation",
            "module": "pygama.dsp.processors",
            "args": ["waveform", "16", "sat_lo", "sat_hi"],
            "unit": "ADC"
        }
    """
    n_lo_out[0] = np.nan
    n_hi_out[0] = np.nan

    if np.isnan(w_in).any() or np.isnan(bit_depth_in):
        return

    if not np.floor(bit_depth_in) == bit_depth_in:
        raise DSPFatal("The bit depth is not an integer")

    if bit_depth_in <= 0:
        raise DSPFatal("The bit depth is not positive")

    n_lo_out[0] = 0
    n_hi_out[0] = 0
    for i in range(0, len(w_in), 1):
        if w_in[i] == 0:
            n_lo_out[0] += 1
        elif w_in[i] == np.power(2, int(bit_depth_in)) - int(bit_depth_in):
            n_hi_out[0] += 1
