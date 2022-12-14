from __future__ import annotations

from typing import Callable

import numpy as np
from numba import guvectorize

from pygama.dsp.errors import DSPFatal
from pygama.dsp.utils import numba_defaults_kwargs as nb_kwargs


def step(length: int) -> Callable:
    """Process waveforms with a step function.

    Note
    ----
    This processor is composed of a factory function that is called using the
    `init_args` argument. The input and output waveforms are passed using
    `args`.

    Parameters
    ----------
    length
        length of the step function.

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "wf_step": {
            "function": "step",
            "module": "pygama.dsp.processors",
            "args": ["waveform", "wf_step(len(waveform)-16+1, 'f')"],
            "unit": "ADC",
            "init_args": ["16"]
        }
    """

    x = np.linspace(0, length, length)
    y = np.piecewise(
        x,
        [
            ((x >= 0) & (x < length / 4)),
            ((x >= length / 4) & (x <= 3 * length / 4)),
            ((x > 3 * length / 4) & (x <= length)),
        ],
        [-1, 1, -1],
    )

    @guvectorize(
        ["void(float32[:], float32[:])", "void(float64[:], float64[:])"],
        "(n),(m)",
        **nb_kwargs(
            cache=False,
            forceobj=True,
        ),
    )
    def step_out(w_in: np.ndarray, w_out: np.ndarray) -> None:
        """
        Parameters
        ----------
        w_in
            the input waveform.
        w_out
            the filtered waveform.
        """

        w_out[:] = np.nan

        if np.isnan(w_in).any():
            return

        if len(y) > len(w_in):
            raise DSPFatal("The filter is longer than the input waveform")
        w_out[:] = np.convolve(w_in, y, mode="valid")

    return step_out


@guvectorize(
    [
        "void(float32[:], float32[:], float32, float32, float32[:])",
        "void(float64[:], float64[:], float64, float64, float64[:])",
    ],
    "(n),(m),(),(),(m)",
    **nb_kwargs,
)
def find_centroid(
    w_in: np.ndarray, idx_in: np.ndarray, length: int, shift: int, idx_out: np.ndarray
) -> None:
    """Calculate centroid position of the waveform.

    Note
    ----
    Processor to find centroid from step output.

    Parameters
    ----------
    w_in
        the input waveform.
    idx_in
        array with input indeces.
    length
        length of the step function.
    shift
        number of sample to shift.
    idx_out
        array with output indeces.

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "idx_out": {
            "function": "find_centroid",
            "module": "pygama.dsp.processors",
            "args": [
                "w_in",
                "idx_in",
                16,
                -15,
                "idx_out"
            ],
            "unit": ["ns"]
        },
    """

    idx_out[:] = np.nan
    k = 0
    for i in range(len(idx_in)):
        if not np.isnan(idx_in[i]):
            t = int(idx_in[i])
            tstart = int(t - length / 2 + length / 10)
            if tstart < 0:
                tstart = 0
            try:
                c_a = np.where(w_in[tstart:t] > 0)[0][0] + tstart
                c_b = np.where(w_in[tstart:t] < 0)[0][-1] + tstart
                idx_out[k] = int(c_a / 2 + c_b / 2) + shift
            except:
                idx_out[k] = 0
            k += 1
