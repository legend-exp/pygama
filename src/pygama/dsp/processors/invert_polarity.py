from __future__ import annotations

import numpy as np
from numba import guvectorize

from pygama.dsp.utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    ["void(float32[:], float32[:])", "void(float64[:], float64[:])"],
    "(n)->(n)",
    **nb_kwargs,
)
def invert_polarity(w_in: np.ndarray, w_out: np.ndarray) -> None:
    """Inverts the polarity of the waveform.

    Parameters
    ----------
    w_in
        the input waveform.
    w_out
        the output inverted waveform.

    JSON Configuration Example
    --------------------------
    
    .. code-block :: json
        "wf_inv": {
            "function": "invert_polarity",
            "module": "pygama.dsp.processors",
            "args": ["wf_pz", "wf_inv"],
            "unit": "ADC"
        }
    """
    w_out[:] = np.nan
    if np.isnan(w_in).any():
        return

    w_out[:] = -w_in[:]
