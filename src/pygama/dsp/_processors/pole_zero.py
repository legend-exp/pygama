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
    constants to the waveform. 

    Parameters
    ----------
    w_in : array-like
        The input waveform
    t_tau1 : float
        The time constant of the first exponential to be deconvolved
    t_tau2 : float
        The time constant of the second exponential to be deconvolved
    frac : float
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

    Notes
    -----
    This algorithm is an IIR filter to deconvolve the function

    .. math:: s(t) = A[frac*\\exp\\left(-\\frac{t}{t_{tau2}}\\right) + (1-frac)*\\exp\\left(-\\frac{t}{t_{tau1}}\\right)]

    into a single step function of amplitude A. 
    This filter is derived by z-transforming the input (:math:`s(t)`) and output (step function) 
    signals, and then determining the transfer function. For shorthand, define :math:`a=\\exp\\left(-\\frac{1}{t_{tau1}}\\right)`
    and :math:`b=\\exp\\left(-\\frac{1}{t_{tau2}}\\right)`, the transfer function is then: 

    .. math:: H(z) =  \\frac{1-(a+b)z^{-1}+abz^{-2}}{1+(fb-fa-b-1)z^{-1}-(fb-fa-b)z^{-2}}

    By equating the transfer function to the ratio of output to input waveforms :math:`H(z) = \\frac{w_{out}(z)}{w_{in}(z)}`
    and then taking the inverse z-transform yields the filter as run in the function, where f is the frac parameter: 

    .. math:: 
        w_{out}[n] = &w_{in}[n]-(a+b)w_{in}[n-1]+(ab)w_{in}[n-2]\\\\
        &-(fb-fa-b-1)w_{out}[n-1]+(fb-fa-b)w_{out}[n-2]
    """
    w_out[:] = np.nan

    if np.isnan(w_in).any() or np.isnan(t_tau1) or np.isnan(t_tau2) or np.isnan(frac):
        return
    if (len(w_in)<=3):
        raise DSPFatal('The length of the waveform must be larger than 3 for the filter to work safely')

    a = np.exp(-1 / t_tau1)
    b = np.exp(-1 / t_tau2)

    transfer_denom_1 = (frac*b-frac*a-b-1)
    transfer_denom_2 = -1*(frac*b-frac*a-b)
    transfer_num_1 = -1*(a+b)
    transfer_num_2 = a*b

    w_out[0] = w_in[0]
    w_out[1] = w_in[1]
    w_out[2] = w_in[2]

    for i in range(2, len(w_in), 1):
        w_out[i] = w_in[i] + transfer_num_1*w_in[i-1] + transfer_num_2*w_in[i-2]\
            - transfer_denom_1*w_out[i-1] - transfer_denom_2*w_out[i-2] 
