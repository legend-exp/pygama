import numpy as np
from numba import guvectorize
from pygama.dsp.errors import DSPFatal

@guvectorize(["void(float32[:], float32, float32[:])",
              "void(float64[:], float64, float64[:])"],
             "(n),()->(n)", nopython=True, cache=True)

def pole_zero(w_in, t_tau, w_out):
    """
    Applies a Pole-zero correction using time constant tau
    Parameters
    ----------
    w_in : array-like
           waveform to apply pole zero correction to. Needs to be baseline subtracted
    
    t_tau : float
            Time constant of exponential decay to be deconvolved
    
    w_out : array-like
            Output array for pole zero corrected waveform 
    Processing Chain Example
    ------------------------
    "wf_pz": {
        "function": "pole_zero",
        "module": "pygama.dsp.processors",
        "args": ["wf_blsub", "db.pz.tau", "wf_pz"],
        "prereqs": ["wf_blsub"],
        "unit": "ADC",
        "defaults": { "db.pz.tau":"74*us" }
        },
    """

    w_out[:] = np.nan 

    if (np.isnan(w_in).any() or np.isnan(t_tau)):
        return

    if (not t_tau > 0):
        raise DSPFatal('t_tau is out of range, must be >= 0')

    const = np.exp(-1/t_tau)
    w_out[0] = w_in[0]
    for i in range(1, len(w_in)):
        w_out[i] = w_out[i-1] + w_in[i] - w_in[i-1] * const


@guvectorize(["void(float32[:], float32, float32, float32, float32[:])",
              "void(float64[:], float64, float64, float64, float64[:])"],
             "(n),(),(),()->(n)", nopython=True, cache=True)

def double_pole_zero(w_in, t_tau1, t_tau2, frac, w_out):
    """
    Pole-zero correction using two time constants: one main (long) time constant
    tau1, and a shorter time constant tau2 that contributes a fraction frac
    Parameters
    ----------
    w_in : array-like
           waveform to apply pole zero correction to. Needs to be baseline subtracted
    
    t_tau1 : float
             Time constant of first exponential decay to be deconvolved
    t_tau2 : float
             Time constant of second exponential decay to be deconvolved
    frac : float
           Fraction which tau2 contributes to decay
    
    w_out : array-like
            Output array for pole zero corrected waveform 
    Processing Chain Example
    ------------------------
    "wf_pz2": {
        "function": "double_pole_zero",
        "module": "pygama.dsp.processors",
        "args": ["wf_blsub", "db.pz2.tau1", "db.pz2.tau2",  "db.pz2.frac", "wf_pz2"],
        "prereqs": ["wf_blsub"],
        "unit": "ADC",
        "defaults": { "db.pz2.tau1":"74*us", "db.pz2.tau2":"3*us", "db.pz2.frac":"0.013" }
        },
    """
    
    w_out[:] = np.nan 

    if (np.isnan(w_in).any() or np.isnan(t_tau1) or np.isnan(t_tau2) or np.isnan(frac)):
        return

    if (not t_tau1 > 0):
        raise DSPFatal('t_tau1 is out of range, must be >= 0')
    if (not t_tau2 > 0):
        raise DSPFatal('t_tau2 is out of range, must be >= 0')
    if (not frac >= 0):
        raise DSPFatal('frac is out of range, must be >= 0')

    const1 = 1/t_tau1 #np.exp(-1/t_tau1)
    const2 = 1/t_tau2 #np.exp(-1/t_tau2)
    w_out[0] = w_in[0]
    e1 = w_in[0]
    e2 = w_in[0]
    e3 = 0
    for i in range(1, len(w_in)):
        e1 += w_in[i] - e2 + e2*const1
        e3 += w_in[i] - e2 - e3*const2
        e2 = w_in[i]
        w_out[i] = e1 - frac*e3
