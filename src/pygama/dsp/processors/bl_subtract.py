from __future__ import annotations

import numpy as np
from numba import guvectorize

from pygama.dsp.utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    ["void(float32[:], float32, float32[:])", "void(float64[:], float64, float64[:])"],
    "(n),()->(n)",
    **nb_kwargs,
)
def bl_subtract(w_in: np.ndarray, a_baseline: float, w_out: np.ndarray) -> None:
    """Subtract the constant baseline from the entire waveform.

    Parameters
    ----------
    w_in
        the input waveform.
    a_baseline
        the baseline value to subtract.
    w_out
        the output waveform with the baseline subtracted.

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "wf_bl": {
            "function": "bl_subtract",
            "module": "pygama.dsp.processors",
            "args": ["waveform", "baseline", "wf_bl"],
            "unit": "ADC"
        }
    """
    w_out[:] = np.nan

    if np.isnan(w_in).any() or np.isnan(a_baseline):
        return

    w_out[:] = w_in[:] - a_baseline
