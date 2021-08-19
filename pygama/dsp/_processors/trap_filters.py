import numpy as np
from numba import guvectorize
from pygama.dsp.errors import DSPFatal

@guvectorize(["void(float32[:], float32, float32, float32[:])",
              "void(float64[:], float64, float64, float64[:])"],
             "(n),(),()->(n)", nopython=True, cache=True)
def trap_filter(w_in, rise, flat, w_out):
    """
    Apply a symmetric trapezoidal filter to the waveform.

    Parameters
    ----------
    w_in : array-like
           The input waveform
    rise : int
           The number of samples averaged in the rise and fall sections
    flat : int
           The delay between the rise and fall sections
    w_out: array-like
           The filtered waveform

    Processing Chain Example
    ------------------------
    "wf_tf": {
        "function": "trap_filter",
        "module": "pygama.dsp.processors",
        "args": ["wf_pz", "10*us", "3*us", "wf_tf"],
        "unit": "ADC",
        "prereqs": ["wf_pz"]
    }
    """
    w_out[:] = np.nan

    if np.isnan(w_in).any() or np.isnan(rise) or np.isnan(flat):
        return

    if np.floor(rise) != rise:
        raise DSPFatal('The number of samples in the rise secion must be an integer')

    if np.floor(flat) != flat:
        raise DSPFatal('The number of samples in the flat secion must be an integer')

    if int(rise) < 0:
        raise DSPFatal('The number of samples in the rise secion must be positive')
    
    if int(flat) < 0:
        raise DSPFatal('The number of samples in the flat secion must be positive')
    
    if 2 * int(rise) + int(flat) > len(w_in):
        raise DSPFatal('The trapezoid width is wider than the waveform')
    
    rise_int = int(rise)
    flat_int = int(flat)
    w_out[0] = w_in[0]
    for i in range(1, rise_int, 1):
        w_out[i] = w_out[i-1] + w_in[i]
    for i in range(rise_int, rise_int + flat_int, 1):
        w_out[i] = w_out[i-1] + w_in[i] - w_in[i-rise_int]
    for i in range(rise_int + flat_int, 2 * rise_int + flat_int, 1):
        w_out[i] = w_out[i-1] + w_in[i] - w_in[i-rise_int] - w_in[i-rise_int-flat_int]
    for i in range(2 * rise_int + flat_int, len(w_in), 1):
        w_out[i] = w_out[i-1] + w_in[i] - w_in[i-rise_int] - w_in[i-rise_int-flat_int] + w_in[i-2*rise_int-flat_int]

@guvectorize(["void(float32[:], float32, float32, float32[:])",
              "void(float64[:], float64, float64, float64[:])"],
             "(n),(),()->(n)", nopython=True, cache=True)
def trap_norm(w_in, rise, flat, w_out):
    """
    Apply a symmetric trapezoidal filter to the waveform, normalized
    by the number of samples averaged in the rise and fall sections.

    Parameters
    ----------
    w_in : array-like
           The input waveform
    rise : int
           The number of samples averaged in the rise and fall sections
    flat : int
           The delay between the rise and fall sections
    w_out: array-like
           The normalized, filtered waveform

    Processing Chain Example
    ------------------------
    "wf_tf": {
        "function": "trap_norm",
        "module": "pygama.dsp.processors",
        "args": ["wf_pz", "10*us", "3*us", "wf_tf"],
        "unit": "ADC",
        "prereqs": ["wf_pz"]
    }
    """
    w_out[:] = np.nan

    if np.isnan(w_in).any() or np.isnan(rise) or np.isnan(flat):
        return

    if np.floor(rise) != rise:
        raise DSPFatal('The number of samples in the rise secion must be an integer')

    if np.floor(flat) != flat:
        raise DSPFatal('The number of samples in the flat secion must be an integer')

    if int(rise) < 0:
        raise DSPFatal('The number of samples in the rise secion must be positive')

    if int(flat) < 0:
        raise DSPFatal('The number of samples in the flat secion must be positive')

    if 2 * int(rise) + int(flat) > len(w_in):
        raise DSPFatal('The trapezoid width is wider than the waveform')
    
    rise_int = int(rise)
    flat_int = int(flat)
    w_out[0] = w_in[0] / rise
    for i in range(1, rise_int, 1):
        w_out[i] = w_out[i-1] + w_in[i] / rise
    for i in range(rise_int, rise_int + flat_int, 1):
        w_out[i] = w_out[i-1] + (w_in[i] - w_in[i-rise_int]) / rise
    for i in range(rise_int + flat_int, 2 * rise_int + flat_int, 1):
        w_out[i] = w_out[i-1] + (w_in[i] - w_in[i-rise_int] - w_in[i-rise_int-flat_int]) / rise
    for i in range(2 * rise_int + flat_int, len(w_in), 1):
        w_out[i] = w_out[i-1] + (w_in[i] - w_in[i-rise_int] - w_in[i-rise_int-flat_int] + w_in[i-2*rise_int-flat_int]) / rise

@guvectorize(["void(float32[:], float32, float32, float32, float32[:])",
              "void(float64[:], float64, float64, float64, float64[:])"],
             "(n),(),(),()->(n)", nopython=True, cache=True)
def asym_trap_filter(w_in, rise, flat, fall, w_out):
    """
    Apply an asymmetric trapezoidal filter to the waveform, normalized
    by the number of samples averaged in the rise and fall sections.

    Parameters
    ----------
    w_in : array-like
           The input waveform
    rise : int
           The number of samples averaged in the rise section
    flat : int
           The delay between the rise and fall sections
    fall : int
           The number of samples averaged in the fall section
    w_out: array-like
           The normalized, filtered waveform

    Processing Chain Example
    ------------------------
    "wf_af": {
        "function": "asym_trap_filter",
        "module": "pygama.dsp.processors",
        "args": ["wf_pz", "128*ns", "64*ns", "2*us", "wf_af"],
        "unit": "ADC",
        "prereqs": ["wf_pz"]
    }
    """
    w_out[:] = np.nan

    if np.isnan(w_in).any() or np.isnan(rise) or np.isnan(flat) or np.isnan(fall):
        return

    if np.floor(rise) != rise:
        raise DSPFatal('The number of samples in the rise secion must be an integer')

    if np.floor(flat) != flat:
        raise DSPFatal('The number of samples in the flat secion must be an integer')

    if np.floor(fall) != fall:
        raise DSPFatal('The number of samples in the fall secion must be an integer')

    if int(rise) < 0:
        raise DSPFatal('The number of samples in the rise secion must be positive')

    if int(flat) < 0:
        raise DSPFatal('The number of samples in the flat secion must be positive')

    if int(fall) < 0:
        raise DSPFatal('The number of samples in the fall secion must be positive')

    if int(rise) + int(flat) + int(fall) > len(w_in):
        raise DSPFatal('The trapezoid width is wider than the waveform')

    rise_int = int(rise)
    flat_int = int(flat)
    fall_int = int(fall)
    w_out[0] = w_in[0] / rise
    for i in range(1, rise_int, 1):
        w_out[i] = w_out[i-1] + w_in[i] / rise
    for i in range(rise_int, rise_int + flat_int, 1):
        w_out[i] = w_out[i-1] + (w_in[i] - w_in[i-rise_int]) / rise
    for i in range(rise_int + flat_int, rise_int + flat_int + fall_int, 1):
        w_out[i] = w_out[i-1] + (w_in[i] - w_in[i-rise_int]) / rise - w_in[i-rise_int-flat_int] / fall
    for i in range(rise_int + flat_int + fall_int, len(w_in), 1):
        w_out[i] = w_out[i-1] + (w_in[i] - w_in[i-rise_int]) / rise - (w_in[i-rise_int-flat_int] - w_in[i-rise_int-flat_int-fall_int]) / fall

@guvectorize(["void(float32[:], float32, float32, float32, float32[:])",
              "void(float64[:], float64, float64, float64, float64[:])"],
             "(n),(),(),()->()", nopython=True, cache=True)
def trap_pickoff(w_in, rise, flat, t_pickoff, a_out):
    """
    Pick off the value at the provided index of a symmetric trapezoidal
    filter to the input waveform, normalized by the number of samples averaged
    in the rise and fall sections.
    
    Parameters
    ----------
    w_in     : array-like
               The input waveform
    rise     : int
               The number of samples averaged in the rise and fall sections
    flat     : int
               The delay between the rise and fall sections
    t_pickoff: int
               The waveform index to pick off
    a_out    : float
               The output pick-off value of the filtered waveform
    
    Processing Chain Example
    ------------------------
    "ct_corr": {
        "function": "trap_pickoff",
        "module": "pygama.dsp.processors",
        "args": ["wf_pz", "1.5*us", 0, "tp_0", "ct_corr"],
        "unit": "ADC",
        "prereqs": ["wf_pz", "tp_0"]
    }
    """
    a_out[0] = np.nan

    if np.isnan(w_in).any() or np.isnan(rise) or np.isnan(flat) or np.isnan(t_pickoff):
        return

    if np.floor(rise) != rise:
        raise DSPFatal('The number of samples in the rise secion must be an integer')

    if np.floor(flat) != flat:
        raise DSPFatal('The number of samples in the flat secion must be an integer')

    if np.floor(t_pickoff) != t_pickoff:
        raise DSPFatal('The pick-off index must be an integer')

    if int(rise) < 0:
        raise DSPFatal('The number of samples in the rise secion must be positive')

    if int(flat) < 0:
        raise DSPFatal('The number of samples in the flat secion must be positive')

    if 2 * int(rise) + int(flat) > len(w_in):
        raise DSPFatal('The trapezoid width is wider than the waveform')

    I_1 = 0.
    I_2 = 0.
    rise_int   = int(rise)
    flat_int   = int(flat)
    start_time = int(t_pickoff + 1)
    for i in range(start_time - rise_int, start_time, 1):
        I_1 += w_in[i]
    for i in range(start_time - 2 * rise_int - flat_int, start_time - rise_int - flat_int, 1):
        I_2 += w_in[i]
    a_out[0] = (I_1 - I_2) / rise
