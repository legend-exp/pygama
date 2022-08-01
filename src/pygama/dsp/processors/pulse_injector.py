from __future__ import annotations

from math import exp, log

import numpy as np
from numba import guvectorize


@guvectorize(["void(float32[:], float32, float32, float32, float32, float32[:])",
              "void(float64[:], float64, float64,float64,float64,float64[:])"],
              "(n),(),(),(), ()->(n)", nopython=True, cache=True)
def inject_sig_pulse(wf_in: np.ndarray, t0: int, rt: float,
                     A: float, decay: float, wf_out: np.ndarray) -> None:
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
    A
        the amplitude :math:`A` of the injected waveform.
    decay
        the decay parameter :math:`\tau` of the injected waveform.
    wf_out
        the output waveform.
    """

    wf_out[:] = np.nan

    if np.isnan(wf_in).any() or np.isnan(rt) or np.isnan(t0) or np.isnan(A) or np.isnan(decay):
        return

    rise = 4*log(99)/rt

    wf_out[:] = wf_in[:]
    for T in range(len(wf_out)):
        wf_out[T] = wf_out[T] + A/(1+exp(-rise*(T-(t0+rt/2))))*exp(-(1/decay)*(T-t0))


@guvectorize(["void(float32[:], float32, float32, float32, float32, float32[:])",
              "void(float64[:], float64, float64,float64,float64,float64[:])"],
              "(n),(),(),(), ()->(n)", nopython=True, cache=True)
def inject_exp_pulse(wf_in: np.ndarray, t0: int, rt: float, A: float,
                     decay: float, wf_out: np.ndarray) -> None:
    """Inject exponential pulse into existing waveform to simulate pileup.

    Parameters
    ----------
    wf_in
        the input waveform.
    t0
        the position of the injected waveform.
    rt
        the rise time of the injected waveform.
    A
        the amplitude of the injected waveform.
    decay
        the exponential decay constant of the injected waveform.
    wf_out
        the output waveform.
    """

    wf_out[:] = np.nan

    if np.isnan(wf_in).any() or np.isnan(rt) or np.isnan(t0) or np.isnan(A) or np.isnan(decay):
        return

    wf_out[:] = wf_in[:]
    for T in range(len(wf_out)):
        if T <= t0 and T <= t0+rt:
            wf_out[T] += (A*exp((T-t0-rt)/(rt))*exp(-(1/decay)*(T-t0)))
        elif (T > t0 + rt):
            wf_out[T] += (A*exp(-(1/decay)*(T-t0)))
