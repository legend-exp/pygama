from __future__ import annotations

import numpy as np
from numba import guvectorize

from pygama.dsp.utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:], float64[:], float64[:])",
    ],
    "(n)->(),(),(),()",
    **nb_kwargs,
)
def min_max(
    w_in: np.ndarray, t_min: int, t_max: int, a_min: float, a_max: float
) -> None:
    """Find the value and index of the minimum and maximum values
    in the waveform.

    Note
    ----
    The first found instance of each extremum in the waveform will be returned.

    Parameters
    ----------
    w_in
        the input waveform
    t_min
        the index of the minimum value
    t_max
        the index of the maximum value
    a_min
        the minimum value
    a_max
        the maximum value

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "tp_min, tp_max, wf_min, wf_max": {
            "function": "min_max",
            "module": "pygama.dsp.processors",
            "args": ["waveform", "tp_min", "tp_max", "wf_min", "wf_max"],
            "unit": ["ns", "ns", "ADC", "ADC"]
        }
    """
    a_min[0] = np.nan
    a_max[0] = np.nan
    t_min[0] = np.nan
    t_max[0] = np.nan

    if np.isnan(w_in).any():
        return

    min_index = 0
    max_index = 0

    for i in range(0, len(w_in), 1):
        if w_in[i] < w_in[min_index]:
            min_index = i
        if w_in[i] > w_in[max_index]:
            max_index = i

    a_min[0] = w_in[min_index]
    a_max[0] = w_in[max_index]
    t_min[0] = float(min_index)
    t_max[0] = float(max_index)
