import numpy as np
from numba import guvectorize

@guvectorize(["void(float32[:], int32, float32, float32[:])",
              "void(float64[:], int64, float64, float64[:])"],
             "(n),(),()->(n)", nopython=True, cache=True)
def soft_pileup_corr(wf_in, n, tau, wf_out):
    """
    Fit the baseline to an exponential with the provided time
    constant and then subtract the best-fit function from the
    entire waveform.
    
    Parameters
    ----------
    wf_in : array-like
            The input waveform
    n     : int
            The number of samples at the beginning of the waveform
            to fit to an exponential
    tau   : float
            The fixed, exponential time constant
    wf_out: array-like
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
    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    s4 = 0.0
    s5 = 0.0
    for i in range(n):
        s1 += 1.0
        s2 += np.exp(-1.0 * i / tau)
        s3 += np.exp(-2.0 * i / tau)
        s4 += np.exp(-1.0 * i / tau) * wf_in[i]
        s5 += wf_in[i]
    B = (s5 - s2 * (s4 * s1 - s2 * s5) / (s3 * s1 - s2 * s2)) / s1
    A = (s4 - B * s2) / s3
    for i, sample in enumerate(wf_in):
        wf_out[i] = sample - (A * np.exp(-1.0 * i / tau) + B)

@guvectorize(["void(float32[:], int32, float32, float32, float32[:])",
              "void(float64[:], int64, float64, float64, float64[:])"],
             "(n),(),(),()->(n)", nopython=True, cache=True)
def soft_pileup_corr_bl(wf_in, n, tau, B, wf_out):
    """
    Fit the baseline to an exponential with the provided time
    constant and then subtract the best-fit function from the
    entire waveform.
    
    Parameters
    ----------
    wf_in : array-like
            The input waveform
    n     : int
            The number of samples at the beginning of the waveform
            to fit to an exponential
    tau   : float
            The fixed, exponential time constant
    B     : float
            The fixed, exponential constant
    wf_out: array-like
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
    s2 = 0.0
    s3 = 0.0
    s4 = 0.0
    for i in range(n):
        s2 += np.exp(-1.0 * i / tau)
        s3 += np.exp(-2.0 * i / tau)
        s4 += np.exp(-1.0 * i / tau) * wf_in[i]
    A = (s4 - B * s2) / s3
    for i, sample in enumerate(wf_in):
        wf_out[i] = sample - (A * np.exp(-1.0 * i / tau) + B)
