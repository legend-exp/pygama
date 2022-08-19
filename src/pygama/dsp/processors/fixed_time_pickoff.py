from __future__ import annotations

import numpy as np
from numba import guvectorize

from pygama.dsp.errors import DSPFatal
from pygama.dsp.utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    ["void(float32[:], float32, float32[:])", "void(float64[:], float64, float64[:])"],
    "(n),()->()",
    **nb_kwargs
)
def fixed_time_pickoff(w_in: np.ndarray, t_in: int, a_out: np.ndarray) -> None:
    """Pick off the waveform value at the provided index.

    If the provided index `t_in` is out of range, return :any:`numpy.nan`.

    Parameters
    ----------
    w_in
        the input waveform
    t_in
        the waveform index to pick off
    a_out
        the output pick-off value

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "trapEftp": {
            "function": "fixed_time_pickoff",
            "module": "pygama.dsp.processors",
            "args": ["wf_trap", "tp_0+10*us", "trapEftp"],
            "unit": "ADC"
        }
    """
    a_out[0] = np.nan

    if np.isnan(w_in).any() or np.isnan(t_in):
        return

    if np.floor(t_in) != t_in:
        raise DSPFatal("The pick-off index must be an integer")

    if int(t_in) < 0 or int(t_in) >= len(w_in):
        return

    a_out[0] = w_in[int(t_in)]
