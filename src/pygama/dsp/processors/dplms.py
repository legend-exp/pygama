from __future__ import annotations

from typing import Callable

import numpy as np
from numba import guvectorize

from pygama.dsp.errors import DSPFatal
from pygama.dsp.utils import numba_defaults_kwargs as nb_kwargs


def dplms(
    noise_mat: list,
    reference: list,
    length: int,
    a1: float,
    a2: float,
    a3: float,
    ff: int,
) -> Callable:
    """Calculate and apply an optimum DPLMS filter to the waveform.

    The processor takes the noise matrix and the reference signal as input and
    calculates the optimum filter according to the provided length and
    penalized coefficients [DPLMS]_.

    .. [DPLMS] V. D'Andrea et al. “Optimum Filter Synthesis with DPLMS
        Method for Energy Reconstruction” In: arXiv (Sept. 2022).
        https://arxiv.org/abs/2209.09863

    Note
    ----
    This processor is composed of a factory function that is called using the
    `init_args` argument. The input and output waveforms are passed using
    `args`.

    Parameters
    ----------
    noise_mat
        noise matrix
    reference
        reference signal
    length
        length of the calculated filter
    a1
        penalized coefficient for the noise matrix
    a2
        penalized coefficient for the reference matrix
    a3
        penalized coefficient for the zero area matrix
    ff
        flat top length for the reference signal


    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "wf_dplms": {
            "function": "dplms",
            "module": "pygama.dsp.processors",
            "args": ["wf_diff", "wf_dplms(len(wf_diff)-49, 'f')"],
            "unit": "ADC",
            "init_args": [
                "db.dplms.noise_matrix",
                "db.dplms.reference",
                "50", "0.1", "1", "0", "1"]
        }
    """

    if length <= 0:
        raise DSPFatal("The length of the filter must be positive")

    noise_mat = np.array(noise_mat)
    reference = np.array(reference)

    if length != noise_mat.shape[0]:
        raise DSPFatal(
            "The length of the filter is not consistent with the noise matrix"
        )

    if len(reference) <= 0:
        raise DSPFatal("The length of the reference signal must be positive")

    if a1 <= 0:
        raise DSPFatal("The penalized coefficient for the noise must be positive")

    if a2 <= 0:
        raise DSPFatal("The penalized coefficient for the reference must be positive")

    if a3 <= 0:
        raise DSPFatal("The penalized coefficient for the zero area must be positive")

    if ff <= 0:
        raise DSPFatal("The penalized coefficient for the ref matrix must be positive")

    # reference matrix
    ssize = len(reference)
    flo = int(ssize / 2 - length / 2)
    fhi = int(ssize / 2 + length / 2)
    ref_mat = np.zeros([length, length])
    ref_sig = np.zeros([length])
    if ff == 0:
        ff = [0]
    elif ff == 1:
        ff = [-1, 0, 1]
    else:
        raise DSPFatal("The penalized coefficient for the ref matrix must be 0 or 1")
    for i in ff:
        ref_mat += np.outer(reference[flo + i : fhi + i], reference[flo + i : fhi + i])
        ref_sig += reference[flo + i : fhi + i]
    ref_mat /= len(ff)

    # filter calculation
    mat = a1 * noise_mat + a2 * ref_mat + a3 * np.ones([length, length])
    x = np.linalg.solve(mat, ref_sig)
    y = np.convolve(reference, np.flip(x), mode="valid")
    maxy = np.amax(y)
    x /= maxy
    y /= maxy

    @guvectorize(
        ["void(float32[:], float32[:])", "void(float64[:], float64[:])"],
        "(n),(m)",
        **nb_kwargs(
            cache=False,
            forceobj=True,
        ),
    )
    def dplms_out(w_in: np.ndarray, w_out: np.ndarray) -> None:
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

        if len(x) > len(w_in):
            raise DSPFatal("The filter is longer than the input waveform")

        w_out[:] = np.convolve(w_in, np.flip(x), "valid")

    return dplms_out
