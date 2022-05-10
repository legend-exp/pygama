import numpy as np
from numba import guvectorize

from pygama.dsp.errors import DSPFatal


@guvectorize(["void(float32[:], float32, float32, float32[:])",
              "void(float64[:], float64, float64, float64[:])"],
             "(n),(),()->(n)", nopython=True, cache=True)
def soft_pileup_corr(w_in, n_in, tau_in, w_out):
    """
    Fit the baseline to an exponential with the provided time
    constant and then subtract the best-fit function from the
    entire waveform.

    Parameters
    ----------
    w_in  : array-like
            The input waveform
    n_in  : int
            The number of samples at the beginning of the waveform
            to fit to an exponential
    tau_in: float
            The fixed, exponential time constant
    w_out : array-like
            The output waveform with the exponential subtracted

    Processing Chain Example
    ------------------------
    "wf_bl": {
        "function": "soft_pileup_corr",
        "module": "pygama.dsp.processors",
        "args": ["waveform", "1000", "500*us", "wf_bl"],
        "unit": "ADC",
        "prereqs": ["waveform"]
    }
    """
    w_out[:] = np.nan

    if np.isnan(w_in).any() or np.isnan(n_in) or np.isnan(tau_in):
        return

    if not np.floor(n_in) == n_in:
        raise DSPFatal('The number of samples is not an integer')

    if n_in < 2:
        raise DSPFatal('The number of samples is not enough for a fit')

    if n_in > len(w_in):
        raise DSPFatal('The number of samples is more than the waveform length')

    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    s4 = 0.0
    s5 = 0.0
    for i in range(0, n_in, 1):
        s1 += 1.0
        s2 += np.exp(-1.0 * i / tau_in)
        s3 += np.exp(-2.0 * i / tau_in)
        s4 += np.exp(-1.0 * i / tau_in) * w_in[i]
        s5 += w_in[i]
    B = (s5 - s2 * (s4 * s1 - s2 * s5) / (s3 * s1 - s2 * s2)) / s1
    A = (s4 - B * s2) / s3
    for i in range(0, len(w_in), 1):
        w_out[i] = w_in[i] - (A * np.exp(-1.0 * i / tau_in) + B)

@guvectorize(["void(float32[:], float32, float32, float32, float32[:])",
              "void(float64[:], float64, float64, float64, float64[:])"],
             "(n),(),(),()->(n)", nopython=True, cache=True)
def soft_pileup_corr_bl(w_in, n_in, tau_in, B_in, w_out):
    """
    Fit the baseline to an exponential with the provided time
    constant and then subtract the best-fit function from the
    entire waveform.

    Parameters
    ----------
    w_in  : array-like
            The input waveform
    n_in  : int
            The number of samples at the beginning of the waveform
            to fit to an exponential
    tau_in: float
            The fixed, exponential time constant
    B_in  : float
            The fixed, exponential constant
    w_out : array-like
            The output waveform with the exponential subtracted

    Processing Chain Example
    ------------------------
    "wf_bl": {
        "function": "soft_pileup_corr_bl",
        "module": "pygama.dsp.processors",
        "args": ["waveform", "1000", "500*us", "baseline", "wf_bl"],
        "unit": "ADC",
        "prereqs": ["waveform", "baseline"]
    }
    """
    w_out[:] = np.nan

    if np.isnan(w_in).any() or np.isnan(n_in) or np.isnan(tau_in) or np.isnan(B_in):
        return

    if not np.floor(n_in) == n_in:
        raise DSPFatal('The number of samples is not an integer')

    if n_in < 1:
        raise DSPFatal('The number of samples is not enough for a fit')

    if n_in > len(w_in):
        raise DSPFatal('The number of samples is more than the waveform length')

    s2 = 0.0
    s3 = 0.0
    s4 = 0.0
    for i in range(0, n_in, 1):
        s2 += np.exp(-1.0 * i / tau_in)
        s3 += np.exp(-2.0 * i / tau_in)
        s4 += np.exp(-1.0 * i / tau_in) * w_in[i]
    A = (s4 - B_in * s2) / s3
    for i in range(0, len(w_in), 1):
        w_out[i] = w_in[i] - (A * np.exp(-1.0 * i / tau_in) + B_in)
