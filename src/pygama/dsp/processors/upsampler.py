from __future__ import annotations

import numpy as np
from numba import guvectorize

from pygama.dsp.errors import DSPFatal
from pygama.dsp.utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    ["void(float32[:], float32, float32[:])", "void(float64[:], float64, float64[:])"],
    "(n),(),(m)",
    **nb_kwargs,
)
def upsampler(w_in: np.ndarray, upsample: float, w_out: np.ndarray) -> None:
    """Upsamples the waveform by the number specified, a series of moving
    windows should be applied afterwards for smoothing.

    Parameters
    ----------
    w_in
        waveform to upsample.
    upsample
        number of samples to increase each sample to.
    w_out
        output array for upsampled waveform.
    """

    w_out[:] = np.nan

    if np.isnan(w_in).any():
        return

    if not (upsample > 0):
        raise DSPFatal("Upsample must be greater than 0")

    for t_in in range(0, len(w_in)):
        t_out = int(t_in * upsample - np.floor(upsample / 2))
        for _ in range(0, int(upsample)):
            if (t_out >= 0) & (t_out < len(w_out)):
                w_out[t_out] = w_in[t_in]
            t_out += 1
