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
def windower(w_in: np.ndarray, t0_in: int, w_out: np.ndarray) -> None:
    """Return a shorter sample of the waveform, starting at the
    specified index.

    Note
    ----
    The length of the output waveform is determined by the length of `w_out`
    rather than an input parameter.  If the the length of `w_out` plus `t0_in`
    extends past the end of `w_in` or if `t0_in` is negative, remaining values
    are padded with :any:`numpy.nan`.

    Parameters
    ----------
    w_in
        the input waveform.
    t0_in
        the starting index of the window.
    w_out
        the windowed waveform.
    """
    w_out[:] = np.nan

    if np.isnan(w_in).any() or np.isnan(t0_in):
        return

    if len(w_out) >= len(w_in):
        raise DSPFatal("The windowed waveform must be smaller than the input waveform")

    beg = min(int(t0_in), len(w_in))
    end = max(beg + len(w_out), 0)
    if beg < 0:
        w_out[: len(w_out) - end] = np.nan
        w_out[len(w_out) - end :] = w_in[:end]
    elif end < len(w_in):
        w_out[:] = w_in[beg:end]
    else:
        w_out[: len(w_in) - beg] = w_in[beg : len(w_in)]
        w_out[len(w_in) - beg :] = np.nan
