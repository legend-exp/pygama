from __future__ import annotations

from math import exp, log

import numpy as np
from numba import guvectorize

from pygama.dsp.utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    [
        "void(float32[:], float32, float32, float32, float32, float32[:])",
        "void(float64[:], float64, float64,float64,float64,float64[:])",
    ],
    "(n),(),(),(), ()->(n)",
    **nb_kwargs,
)
def inject_sig_pulse(
    wf_in: np.ndarray, t0: int, rt: float, a: float, decay: float, wf_out: np.ndarray
) -> None:
    r"""Inject sigmoid pulse into existing waveform to simulate pileup.

    .. math::
        s(t) = \frac{A}{1 + \exp[-4 \log(99) (t - t_0 - t_r/2) / t_r]}
                e^{-(t-t_0)/\tau}

    Parameters
    ----------
    wf_in
        the input waveform.
    t0
        the position :math:`t_0` of the injected waveform.
    rt
        the rise time :math:`t_r` of the injected waveform.
    a
        the amplitude :math:`A` of the injected waveform.
    decay
        the decay parameter :math:`\tau` of the injected waveform.
    wf_out
        the output waveform.
    """

    wf_out[:] = np.nan

    if (
        np.isnan(wf_in).any()
        or np.isnan(rt)
        or np.isnan(t0)
        or np.isnan(a)
        or np.isnan(decay)
    ):
        return

    rise = 4 * log(99) / rt

    wf_out[:] = wf_in[:]
    for t in range(len(wf_out)):
        wf_out[t] = wf_out[t] + a / (1 + exp(-rise * (t - (t0 + rt / 2)))) * exp(
            -(1 / decay) * (t - t0)
        )


@guvectorize(
    [
        "void(float32[:], float32, float32, float32, float32, float32[:])",
        "void(float64[:], float64, float64,float64,float64,float64[:])",
    ],
    "(n),(),(),(), ()->(n)",
    **nb_kwargs,
)
def inject_exp_pulse(
    wf_in: np.ndarray, t0: int, rt: float, a: float, decay: float, wf_out: np.ndarray
) -> None:
    """Inject exponential pulse into existing waveform to simulate pileup.

    Parameters
    ----------
    wf_in
        the input waveform.
    t0
        the position of the injected waveform.
    rt
        the rise time of the injected waveform.
    a
        the amplitude of the injected waveform.
    decay
        the exponential decay constant of the injected waveform.
    wf_out
        the output waveform.
    """

    wf_out[:] = np.nan

    if (
        np.isnan(wf_in).any()
        or np.isnan(rt)
        or np.isnan(t0)
        or np.isnan(a)
        or np.isnan(decay)
    ):
        return

    wf_out[:] = wf_in[:]
    for t in range(len(wf_out)):
        if t <= t0 and t <= t0 + rt:
            wf_out[t] += a * exp((t - t0 - rt) / (rt)) * exp(-(1 / decay) * (t - t0))
        elif t > t0 + rt:
            wf_out[t] += a * exp(-(1 / decay) * (t - t0))
