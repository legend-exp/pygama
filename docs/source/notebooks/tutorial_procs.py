import numpy as np
from numba import guvectorize
from scipy.ndimage import gaussian_filter1d

from pygama.dsp.errors import DSPFatal
from pygama.dsp.utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    ["void(float32[:], float32, float32[:])", "void(float64[:], float64, float64[:])"],
    "(n),()->(n)",
    **nb_kwargs,
)
def pole_zero(w_in: np.ndarray, t_tau: float, w_out: np.ndarray) -> None:
    """Apply a pole-zero cancellation using the provided time
    constant to the waveform.

    Parameters
    ----------
    w_in
        the input waveform.
    t_tau
        the time constant of the exponential to be deconvolved.
    w_out
        the pole-zero cancelled waveform.

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "wf_pz": {
            "function": "pole_zero",
            "module": "dsp_tutorial",
            "args": ["wf_bl", "400*us", "wf_pz"],
            "unit": "ADC"
        }
    """
    if np.isnan(t_tau) or t_tau == 0:
        raise DSPFatal("t_tau must be a non-zero number")

    w_out[:] = np.nan

    if np.isnan(w_in).any():
        return

    const = np.exp(-1 / t_tau)
    w_out[0] = w_in[0]
    for i in range(1, len(w_in), 1):
        w_out[i] = w_out[i - 1] + w_in[i] - w_in[i - 1] * const


@guvectorize(
    ["(float32[:], float32[:])", "(float64[:], float64[:])"], "(n),(m)", **nb_kwargs
)
def derivative(w_in: np.ndarray, w_out: np.ndarray):
    """Calculate time-derivative of pulse by taking finite
    difference across n_samples points, where n_samples is
    `len(w_in) - len(w_out)`.

    Parameters
    ----------
    w_in
        the input waveform.
    w_out
        the derivative waveform.

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "wf_deriv": {
            "function": "derivative",
            "module": "dsp_tutorial",
            "args": ["wf_in", "wf_deriv(shape=len(wf_in)-5)"],
            "unit": "ADC/us"
        }
    """
    n_samp = len(w_in) - len(w_out)
    if n_samp < 0:
        raise DSPFatal("n_samples must be >0")

    w_out[:] = np.nan

    if np.isnan(w_in).any():
        return

    for i_samp in range(len(w_out)):
        w_out[i_samp] = w_in[i_samp + n_samp] - w_in[i_samp]


@guvectorize(
    ["(float32[:], float32, float32[:])", "(float64[:], float64, float64[:])"],
    "(n),()->(n)",
    **nb_kwargs(forceobj=True),
)
def gauss_filter(w_in: np.ndarray, sigma: float, w_out: np.ndarray):
    """Convolve the waveform with a gaussian function

    Parameters
    ----------
    w_in
        the input waveform.
    sigma
        the standard deviation of the Gaussian filter
    w_out
        the derivative waveform.

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "wf_gauss": {
            "function": "gauss_filter",
            "module": "dsp_tutorial",
            "args": ["wf_in", "100*ns", "wf_gauss"],
            "unit": "ADC"
        }
    """

    w_out[:] = np.nan

    if np.isnan(w_in).any():
        return

    gaussian_filter1d(w_in, sigma, output=w_out, mode="nearest")


def triangle_filter(length: int):
    """Convolve the waveform with a triangle function

    Parameters
    ----------
    length
        the total number of samples for the triangle kernel

    Returns
        gufunc for triangle filter

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "wf_tri": {
            "function": "triangle_filter",
            "module": "dsp_tutorial",
            "args": ["wf_in", "wf_gauss"],
            "init_args": ["100*ns"]
            "unit": "ADC"
        }
    """

    # build triangular kernel
    kernel = np.concatenate(
        [
            np.arange(1, length // 2 + 1, dtype="f"),
            np.arange((length + 1) // 2, 0, -1, dtype="f"),
        ]
    )
    kernel /= np.sum(kernel)  # normalize

    @guvectorize(["(float32[:], float32[:])"], "(n)->(n)", forceobj=True, cache=False)
    def returned_filter(w_in: np.ndarray, w_out: np.ndarray):
        w_out[:] = np.nan

        if np.isnan(w_in).any():
            return

        w_out[:] = np.convolve(w_in, kernel, mode="same")

    return returned_filter
