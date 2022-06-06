import numpy as np
from numba import guvectorize

from pygama.dsp.errors import DSPFatal


@guvectorize(["void(float32[:], float32, float32[:])",
              "void(float64[:], float64, float64[:])"],
             "(n),()->(n)", nopython=True, cache=True)
def pole_zero(w_in, t_tau, w_out):
    """
    Apply a pole-zero cancellation using the provided time
    constant to the waveform.

    Parameters
    ----------
    w_in : array-like
        The input waveform
    t_tau : float
        The time constant of the exponential to be deconvolved
    w_out : array-like
        The pole-zero cancelled waveform

    Examples
    --------
    .. code-block :: json

        "wf_pz": {
            "function": "pole_zero",
            "module": "pygama.dsp.processors",
            "args": ["wf_bl", "400*us", "wf_pz"],
            "unit": "ADC",
            "prereqs": ["wf_bl"]
        }
    """
    w_out[:] = np.nan

    if np.isnan(w_in).any() or np.isnan(t_tau):
        return

    const = np.exp(-1 / t_tau)
    w_out[0] = w_in[0]
    for i in range(1, len(w_in), 1):
        w_out[i] = w_out[i-1] + w_in[i] - w_in[i-1] * const

@guvectorize(["void(float32[:], float32, float32, float32, float32[:])",
              "void(float64[:], float64, float64, float64, float64[:])"],
             "(n),(),(),()->(n)", nopython=True, cache=True)
def double_pole_zero(w_in, t_tau1, t_tau2, frac, w_out):
    """
    Apply a double pole-zero cancellation using the provided time
    constant to the waveform. This algorithm is a IIR filter to deconvolve the function
    s(x) = A[frac*exp(-t/t_tau2) + (1-frac)*exp(-t/t_tau1)] into a single step function
    of amplitude A. See the June 1st analysis and simulations call for more details. 
    https://indico.legend-exp.org/event/840/

    Parameters
    ----------
    w_in  : array-like
            The input waveform
    t_tau1: float
            The time constant of the first exponential to be deconvolved
    t_tau2: float
            The time constant of the second exponential to be deconvolved
    frac  : float
            The fraction which the second exponential contributes
    w_out : array-like
            The pole-zero cancelled waveform

    Examples
    --------
    .. code-block :: json

        "wf_pz": {
            "function": "double_pole_zero",
            "module": "pygama.dsp.processors",
            "args": ["wf_bl", "400*us", "20*us", "0.02", "wf_pz"],
            "unit": "ADC",
            "prereqs": ["wf_bl"]
        }
    """
    w_out[:] = np.nan 

    if np.isnan(w_in).any() or np.isnan(t_tau1) or np.isnan(t_tau2) or np.isnan(frac):
        return 
    if (len(w_in)<=3):
        raise DSPFatal('The length of the waveform must be larger than 3 for the filter to work safely')

    a = np.exp(-1 / t_tau1)
    b = np.exp(-1 / t_tau2)
    w_out[0] = w_in[0]
    w_out[1] = w_in[1]
    w_out[2] = w_in[2]

    for i in range(2, len(w_in), 1):
        w_out[i] = -(frac*b-frac*a-b-1)*w_out[i-1] + (frac*b-frac*a-b)*w_out[i-2] + w_in[i] - (a+b)*w_in[i-1] + (a*b)*w_in[i-2]

