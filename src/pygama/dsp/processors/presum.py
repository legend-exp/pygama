from __future__ import annotations

import numpy as np
from numba import guvectorize


@guvectorize(["void(float32[:], float32[:])",
              "void(float64[:], float64[:])"],
             "(n),(m)", nopython=True, cache=True)
def presum(w_in: np.ndarray, w_out: np.ndarray) -> None:
    """Presum the waveform.

    Combine bins in chunks of ``len(w_in) / len(w_out)``, which is hopefully an
    integer. If it isn't, then some samples at the end will be omitted.

    Parameters
    ----------
    w_in
        the input waveform.
    w_out
        the output waveform.
    """
    w_out[:] = np.nan

    if np.isnan(w_in).any():
        return

    ps_fact = len(w_in) // len(w_out)
    for i in range(0, len(w_out), 1):
        j0 = i * ps_fact
        w_out[i] = w_in[j0]/ps_fact
        for j in range(j0 + 1, j0 + ps_fact, 1):
            w_out[i] += w_in[j]/ps_fact
