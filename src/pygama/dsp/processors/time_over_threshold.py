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
    if np.isnan(w_in).any() or np.isnan(a_threshold):
        n_samples[0] = np.nan
        return
    n_samples[0] = 0.0
    for sample in w_in[:]:
        if sample > a_threshold:
            n_samples[0] += 1
