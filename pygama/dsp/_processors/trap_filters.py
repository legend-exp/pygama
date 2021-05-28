import numpy as np
from numba import guvectorize
import math
from pygama.dsp.errors import DSPFatal


@guvectorize(["void(float32[:], float32, float32, float32[:])",
              "void(float64[:], float32, float32, float64[:])"],
             "(n),(),()->(n)", nopython=True, cache=True)
def trap_filter(w_in, rise, flat, w_out):
    
    """
    Applies a symmetric trapezoidal filter (rise= fall) to the waveform 
    Parameters:
    -----------
    w_in : array-like
           Input Waveform
    rise : float
           Sets the number of samples that will be averaged in the rise and fall sections
    flat : float
            Controls the delay between the rise and fall averaging sections, 
            typically around 3us for ICPC energy estimation, lower for detectors with shorter drift times
    w_out : array-like
            Output waveform after trap filter applied
    Processing Chain Example
    ------------------------
    "wf_trap": {
        "function": "trap_filter",
        "module": "pygama.dsp.processors",
        "args": ["wf_pz", "10*us", "3*us", "wf_trap"],
        "prereqs": ["wf_pz"],
        "unit": "ADC"
        },
    """
    
    w_out[:] = np.nan

    if (np.isnan(w_in).any()):
        return

    if (not  0 <= rise):
        raise DSPFatal('Rise must be >= 0')
    
    if (not  0 <= flat):
        raise DSPFatal('Flat must be >= 0')
    
    if (not 2*rise+flat <= len(w_in)):
        raise DSPFatal('Trap Filter longer than waveform')
    
    rise_int = int(rise)
    flat_int = int(flat)
    w_out[0] = w_in[0]
    for i in range(1, rise_int):
        w_out[i] = w_out[i-1] + w_in[i]
    for i in range(rise_int, rise_int+flat_int):
        w_out[i] = w_out[i-1] + w_in[i] - w_in[i-rise_int]
    for i in range(rise_int+flat_int, 2*rise_int+flat_int):
        w_out[i] = w_out[i-1] + w_in[i] - w_in[i-rise_int] - w_in[i-rise_int-flat_int]
    for i in range(2*rise_int+flat_int, len(w_in)):
        w_out[i] = w_out[i-1] + w_in[i] - w_in[i-rise_int] - w_in[i-rise_int-flat_int] + w_in[i-2*rise_int-flat_int]

@guvectorize(["void(float32[:], float32, float32, float32[:])",
              "void(float64[:], float32, float32, float64[:])"],
             "(n),(),()->(n)", nopython=True, cache=True)

def trap_norm(w_in, rise, flat, w_out):
   
    """
    Applies a symmetric trapezoidal filter (rise= fall) to the waveform normalized by integration time
    Parameters:
    -----------
    w_in : array-like
           Input Waveform
    rise : float
           Sets the number of samples that will be averaged in the rise and fall sections
    flat : float
            Controls the delay between the rise and fall averaging sections, 
            typically around 3us for ICPC energy estimation, lower for detectors with shorter drift times
    w_out : array-like
            Output waveform after trap filter applied
    Processing Chain Example
    ------------------------
    "wf_trap": {
        "function": "trap_norm",
        "module": "pygama.dsp.processors",
        "args": ["wf_pz", "10*us", "3*us", "wf_trap"],
        "prereqs": ["wf_pz"],
        "unit": "ADC"
        },
    """
    
    w_out[:] = np.nan

    if (np.isnan(w_in).any()):
        return

    if (not  0 <= rise):
        raise DSPFatal('Rise must be >= 0')
    
    if (not  0 <= flat):
        raise DSPFatal('Flat must be >= 0')
    
    if (not 2*rise+flat <= len(w_in)):
        raise DSPFatal('Trap Filter longer than waveform')
    
    rise_int = int(rise)
    flat_int = int(flat)
    w_out[0] = w_in[0]/float(rise)
    for i in range(1, rise_int):
        w_out[i] = w_out[i-1] + w_in[i]/rise
    for i in range(rise_int, rise_int+flat_int):
        w_out[i] = w_out[i-1] + (w_in[i] - w_in[i-rise_int])/rise
    for i in range(rise_int+flat_int, 2*rise_int+flat_int):
        w_out[i] = w_out[i-1] + (w_in[i] - w_in[i-rise_int] - w_in[i-rise_int-flat_int])/rise
    for i in range(2*rise_int+flat_int, len(w_in)):
        w_out[i] = w_out[i-1] + (w_in[i] - w_in[i-rise_int] - w_in[i-rise_int-flat_int] + w_in[i-2*rise_int-flat_int])/rise

@guvectorize(["void(float32[:], float32, float32, float32, float32[:])",
              "void(float64[:], float32, float32, float32, float64[:])"],
             "(n),(),(),()->(n)", nopython=True, cache=True)

def asym_trap_filter(w_in, rise, flat, fall, w_out):
       
    """
    Applies asymmetric trapezoidal filter to the waveform normalized by the integration times
    Parameters:
    ----------
    w_in : array-like
           Input Waveform
    rise : float
           Sets the number of samples that will be averaged in the rise section
    flat : float
            Controls the delay between the rise and fall averaging sections, 
            typically around 3us for ICPC energy estimation, lower for detectors with shorter drift times
    rise : float
           Sets the number of samples that will be averaged in the fall section
    w_out : array-like
            Output waveform after trap filter applied
    Processing Chain Example
    ------------------------
    "wf_atrap": {
        "function": "asym_trap_filter",
        "module": "pygama.dsp.processors",
        "args": ["wf_pz", "128*ns", "64*ns","2*us", "wf_atrap"],
        "prereqs": ["wf_pz"],
        "unit": "ADC"
        },
    """

    w_out[:] = np.nan

    if (np.isnan(w_in).any()):
        return

    if (not  0 <= rise):
        raise DSPFatal('Rise must be >= 0')
    
    if (not  0 <= flat):
        raise DSPFatal('Flat must be >= 0')

    if (not  0 <= fall):
        raise DSPFatal('Fall must be >= 0')
    
    if (not rise+flat+fall <= len(w_in)):
        raise DSPFatal('Trap Filter longer than waveform')

    rise_int = int(rise)
    flat_int = int(flat)
    fall_int = int(fall)
    w_out[0] = w_in[0]/float(rise)
    for i in range(1, rise_int):
        w_out[i] = w_out[i-1] + w_in[i]/rise
    for i in range(rise_int, rise_int+flat_int):
        w_out[i] = w_out[i-1] + (w_in[i] - w_in[i-rise_int])/rise
    for i in range(rise_int+flat_int, rise_int+flat_int+fall_int):
        w_out[i] = w_out[i-1] + (w_in[i] - w_in[i-rise_int])/rise - w_in[i-rise_int-flat_int]/fall
    for i in range(rise_int+flat_int+fall_int, len(w_in)):
        w_out[i] = w_out[i-1] + (w_in[i] - w_in[i-rise_int])/rise - (w_in[i-rise_int-flat_int] - w_in[i-rise_int-flat_int-fall_int])/fall

@guvectorize(["void(float32[:], float32, float32, float32, float32[:])",
              "void(float64[:], float32, float32, float32, float64[:])"],
             "(n),(),(),()->()", nopython=True, cache=True)

def trap_pickoff(w_in, rise, flat, t_pickoff, a_out):
    """
    Gives the value of a trapezoid, normalized by the "rise" (integration) time, at a specific time equal to pickoff_time. Use when the rest of the trapezoid output is extraneous.
    
    Parameters:
    ----------
    w_in : array-like
           Input Waveform
    rise : float
           Sets the number of samples that will be averaged in the rise section
    flat : float
            Controls the delay between the rise and fall averaging sections, 
            typically around 3us for ICPC energy estimation, lower for detectors with shorter drift times
    t_pickoff : float
            Time to take sample
    a_out : float
            Output waveform after trap filter applied
    
    Processing Chain Example
    ------------------------
    "ct_corr": {
        "function": "trap_pickoff",
        "module": "pygama.dsp.processors",
        "args":["wf_pz", "1.5*us", 0, "tp_0", "ct_corr"],
        "unit": "ADC",
        "prereqs": ["wf_pz", "tp_0"]
        },
    
    """

    a_out[0] = np.nan

    if (np.isnan(w_in).any()):
        return

    if (not  0 <= t_pickoff <= len(w_in)) :
        return

    if (not np.floor(t_pickoff)==t_pickoff):
        raise DSPFatal('Pickoff time is not an integer')

    if (not  0 <= rise):
        raise DSPFatal('Rise must be >= 0')
    
    if (not  0 <= flat):
        raise DSPFatal('Flat must be >= 0')

    if (not 2*rise+flat <= len(w_in)):
        raise DSPFatal('Trap Filter longer than waveform')


    I_1 = 0.
    I_2 = 0.
    rise_int = int(rise)
    flat_int = int(flat)
    start_time = int(t_pickoff + 1) # the +1 makes slicing prettier
    
    for i in range(start_time-rise_int, start_time):
        I_1 += w_in[i]

    for k in range(start_time - 2*rise_int - flat_int, start_time - rise_int - flat_int):
        I_2 += w_in[k]

    a_out[0] = (I_1 - I_2)/rise
