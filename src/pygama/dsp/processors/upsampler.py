from __future__ import annotations

from math import ceil, floor

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
    """Upsamples the waveform by the number specified.

    Note
    ----
    A series of moving windows should be applied afterwards for smoothing.

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


@guvectorize(
    ["void(float32[:], char, float32[:])", "void(float64[:], char, float64[:])"],
    "(n),(),(m)",
    nopython=True,
    **nb_kwargs,
)
def interpolating_upsampler(
    w_in: np.ndarray, mode_in: np.int8, w_out: np.ndarray
) -> None:
    """Upsamples the waveform.

    Resampling ratio is set by size of `w_out` and `w_in`. Using the
    interpolation technique specified by `mode_in` to fill newly added samples.

    Parameters
    ----------
    w_in
        waveform to upsample.
    mode_in
        character selecting which interpolation method to use. Note this
        must be passed as a ``int8``, e.g. ``ord('i')``. Options:

        * ``i`` -- only set values at original samples; fill in between
          with zeros. Requires integer resampling ratio
        * ``n`` -- nearest-neighbor interpolation; defined at all values,
          but not continuous
        * ``f`` -- floor, or value at previous neighbor; defined at all
          values but not continuous
        * ``c`` -- ceiling, or value at next neighbor; defined at all values,
          but not continuous
        * ``l`` -- linear interpolation; continuous at all values, but not
          differentiable
        * ``h`` -- Hermite cubic spline interpolation; continuous and
          differentiable at all values but not twice-differentiable
        * ``s`` -- natural cubic spline interpolation; continuous and twice-
          differentiable at all values. This method is much slower
          than the others because it utilizes the entire input waveform!
    w_out
        output array for upsampled waveform.

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "wf_up": {
            "function": "interpolating_upsampler",
            "module": "pygama.dsp.processors",
            "args": ["wf", "'s'", "wf_up(len(wf)*10, period=wf.period/10)"],
            "unit": "ADC"
        }
    """

    w_out[:] = np.nan

    if np.isnan(w_in).any():
        return

    upsample = len(w_out) / len(w_in)

    if mode_in == ord("i"):  # Index
        if upsample != int(upsample):
            raise DSPFatal(
                "interpolating_upsampler requires len(w_out) to be an integer multiple of len(w_in) for mode 'i'"
            )
        for i_in, a in enumerate(w_in):
            i_out = int(upsample * i_in)
            w_out[i_out] = a
            w_out[i_out + 1 : i_out + upsample] = 0

    elif mode_in == ord("n"):  # Nearest-neighbor
        i_last = 0
        for i_in, a in enumerate(w_in):
            i_next = ceil(upsample * (i_in + 0.5))
            w_out[i_last:i_next] = a
            i_last = i_next
        w_out[i_last:] = w_in[-1]

    elif mode_in == ord("f"):  # Floor
        i_last = 0
        for i_in, a in enumerate(w_in):
            i_next = ceil(upsample * (i_in + 1))
            w_out[i_last:i_next] = a
            i_last = i_next

    elif mode_in == ord("c"):  # Ceiling
        i_last = 0
        for i_in, a in enumerate(w_in):
            i_next = floor(upsample * (i_in)) + 1
            w_out[i_last:i_next] = a
            i_last = i_next
        w_out[i_last:] = w_in[-1]

    elif mode_in == ord("l"):  # linear
        i_last = 0
        for i_in, a in enumerate(w_in):
            i_next = ceil(upsample * (i_in + 1))
            a_next = w_in[i_in + 1] if i_in < len(w_in) - 1 else w_in[-1]
            for i_out in range(i_last, i_next):
                t = i_out / upsample - i_in
                w_out[i_out] = a + t * (a_next - a)
            i_last = i_next

    elif mode_in == ord("h"):  # Cubic hermite
        i_last = 0
        for i_in in range(len(w_in) - 1):
            i_next = ceil(upsample * (i_in + 1))
            m0 = (
                w_in[1] - w_in[0]
                if i_in == 0
                else (w_in[i_in + 1] - w_in[i_in - 1]) / 2
            )
            m1 = (
                w_in[-1] - w_in[-2]
                if i_in == len(w_in) - 2
                else (w_in[i_in + 2] - w_in[i_in]) / 2
            )
            for i_out in range(i_last, i_next):
                t0 = i_out / upsample - i_in
                t1 = 1 - t0
                w_out[i_out] = (
                    (-2 * t1**3 + 3 * t1**2) * w_in[i_in]
                    + (-2 * t0**3 + 3 * t0**2) * w_in[i_in + 1]
                    - (t1**3 - t1**2) * m0
                    + (t0**3 - t0**2) * m1
                )
            i_last = i_next

        for i_out in range(i_last, len(w_out)):
            t0 = i_out / upsample - i_in
            t1 = 1 - t0
            w_out[i_out] = (
                (-2 * t1**3 + 3 * t1**2) * w_in[i_in]
                + (-2 * t0**3 + 3 * t0**2) * w_in[i_in + 1]
                - (t1**3 - t1**2) * m0
                + (t0**3 - t0**2) * m1
            )

    elif mode_in == ord("s"):  # Cubic spline
        u = np.zeros(len(w_in))
        w2 = np.zeros(len(w_in))  # second derivative values

        for i_in in range(1, len(w_in) - 1):
            p = 0.5 * w2[i_in - 1] + 2
            w2[i_in] = -0.5 / p
            u[i_in] = w_in[i_in + 1] - 2 * w_in[i_in] + w_in[i_in - 1]
            u[i_in] = (3 * u[i_in] - 0.5 * u[i_in - 1]) / p

        i_last = len(w_out)
        for i_in in range(len(w_in) - 2, -1, -1):
            w2[i_in] = w2[i_in] * w2[i_in + 1] + u[i_in]
            i_next = ceil(upsample * (i_in))

            for i_out in range(i_last, i_next - 1, -1):
                t0 = i_out / upsample - i_in
                t1 = 1 - t0
                w_out[i_out] = (
                    t1 * w_in[i_in]
                    + t0 * w_in[i_in + 1]
                    + ((t1**3 - t1) * w2[i_in] + (t0**3 - t0) * w2[i_in + 1]) / 6.0
                )
            i_last = i_next

    else:
        raise DSPFatal("Unrecognized interpolation mode")
