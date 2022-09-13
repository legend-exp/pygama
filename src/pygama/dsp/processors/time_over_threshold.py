from __future__ import annotations

import numpy as np
from numba import guvectorize

from pygama.dsp.utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    ["void(float32[:], float32, float32[:])", "void(float64[:], float64, float64[:])"],
    "(n),()->()",
    **nb_kwargs,
)
def time_over_threshold(w_in: np.ndarray, a_threshold: float, n_samples: float) -> None:
    """Calculates the number of samples in the input waveform over a_threshold

    Parameters
    ----------
    w_in
        the input waveform.
    a_threshold
        the threshold value.
    n_samples
        the number of samples over the threshold.

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "t_sat": {
            "function": "time_over_threshold",
            "module": "pygama.dsp.processors",
            "args": ["wf_pz", "a_threshold", "t_sat"],
            "unit": "ns"
        }
    """
    if np.isnan(w_in).any() or np.isnan(a_threshold):
        n_samples[0] = np.nan
        return
    n_samples[0] = 0.0
    for sample in w_in[:]:
        if sample > a_threshold:
            n_samples[0] += 1
