from __future__ import annotations

import numpy as np
from numba import guvectorize

from pygama.dsp.errors import DSPFatal
from pygama.dsp.utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    [
        "void(float32[:], float32, float32, float32, float32[:])",
        "void(float64[:], float64, float64, float64, float64[:])",
    ],
    "(n),(),(),()->()",
    **nb_kwargs,
)
def time_point_thresh(
    w_in: np.ndarray, a_threshold: float, t_start: int, walk_forward: int, t_out: float
) -> None:
    """Find the index where the waveform value crosses the threshold, walking
    either forward or backward from the starting index.

    Parameters
    ----------
    w_in
        the input waveform.
    a_threshold
        the threshold value.
    t_start
        the starting index.
    walk_forward
        the backward (``0``) or forward (``1``) search direction.
    t_out
        the index where the waveform value crosses the threshold.

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "tp_0": {
            "function": "time_point_thresh",
            "module": "pygama.dsp.processors",
            "args": ["wf_atrap", "bl_std", "tp_start", 0, "tp_0"],
            "unit": "ns"
        }
    """
    t_out[0] = np.nan

    if (
        np.isnan(w_in).any()
        or np.isnan(a_threshold)
        or np.isnan(t_start)
        or np.isnan(walk_forward)
    ):
        return

    if np.floor(t_start) != t_start:
        raise DSPFatal("The starting index must be an integer")

    if np.floor(walk_forward) != walk_forward:
        raise DSPFatal("The search direction must be an integer")

    if int(t_start) < 0 or int(t_start) >= len(w_in):
        raise DSPFatal("The starting index is out of range")

    if int(walk_forward) == 1:
        for i in range(int(t_start), len(w_in) - 1, 1):
            if w_in[i] <= a_threshold < w_in[i + 1]:
                t_out[0] = i
                return
    else:
        for i in range(int(t_start), 1, -1):
            if w_in[i - 1] < a_threshold <= w_in[i]:
                t_out[0] = i
                return


@guvectorize(
    [
        "void(float32[:], float32, float32, int64, char, float32[:])",
        "void(float64[:], float64, float64, int64, char, float64[:])",
    ],
    "(n),(),(),(),()->()",
    **nb_kwargs,
)
def interpolated_time_point_thresh(
    w_in: np.ndarray,
    a_threshold: float,
    t_start: int,
    walk_forward: int,
    mode_in: np.int8,
    t_out: float,
) -> None:
    """Find the time where the waveform value crosses the threshold

    Search performed walking either forward or backward from the starting
    index. Use interpolation to estimate a time between samples. Interpolation
    mode selected with `mode_in`.

    Parameters
    ----------
    w_in
        the input waveform.
    a_threshold
        the threshold value.
    t_start
        the starting index.
    walk_forward
        the backward (``0``) or forward (``1``) search direction.
    mode_in
        Character selecting which interpolation method to use. Note this
        must be passed as a ``int8``, e.g. ``ord('i')``. Options:

        * ``i`` -- integer `t_in`; equivalent to
          :func:`~.dsp.processors.fixed_sample_pickoff`
        * ``f`` -- floor; interpolated values are at previous neighbor, so
          threshold crossing is at next neighbor
        * ``c`` -- ceiling, interpolated values are at next neighbor, so
          threshold crossing is at previous neighbor
        * ``n`` -- nearest-neighbor interpolation; threshold crossing is
          half-way between samples
        * ``l`` -- linear interpolation
        * ``h`` -- Hermite cubic spline interpolation (*not implemented*)
        * ``s`` -- natural cubic spline interpolation (*not implemented*)
    t_out
        the index where the waveform value crosses the threshold.

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "tp_0": {
            "function": "time_point_thresh",
            "module": "pygama.dsp.processors",
            "args": ["wf_atrap", "bl_std", "tp_start", 0, "'l'", "tp_0"],
            "unit": "ns"
        }
    """
    t_out[0] = np.nan

    if (
        np.isnan(w_in).any()
        or np.isnan(a_threshold)
        or np.isnan(t_start)
        or np.isnan(walk_forward)
    ):
        return

    if t_start < 0 or t_start >= len(w_in):
        return

    i_cross = -1
    if walk_forward > 0:
        for i in range(int(t_start), len(w_in) - 1, 1):
            if w_in[i] <= a_threshold < w_in[i + 1]:
                i_cross = i
    else:
        for i in range(int(t_start), 1, -1):
            if w_in[i - 1] < a_threshold <= w_in[i]:
                i_cross = i - 1

    if i_cross == -1:
        return

    if mode_in == ord("i"):  # return index before crossing
        t_out[0] = i_cross
    elif mode_in == ord("f"):  # return index before crossing
        t_out[0] = i_cross + 1
    elif mode_in == ord("c"):  # return index before crossing
        t_out[0] = i_cross
    elif mode_in == ord("n"):  # nearest-neighbor; return half-way between samps
        t_out[0] = i_cross + 0.5
    elif mode_in == ord("l"):  # linear
        t_out[0] = i_cross + (a_threshold - w_in[i_cross]) / (
            w_in[i_cross + 1] - w_in[i_cross]
        )
    else:
        raise DSPFatal("Unrecognized interpolation mode")
