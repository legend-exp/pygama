from numba import guvectorize
import numpy as np
from pygama.dsp.errors import DSPFatal

@guvectorize(["void(float32[:], float64, float32[:])",
              "void(float64[:], float64, float64[:])"],
             "(n),(),(m)", nopython=True, cache=True)
def windower(wf_in, t0_in, wf_out):
    """Waveform windower: returns a shorter sample of the waveform
    starting at t0_in. Note that the length of the output waveform
    is determined by the length of wf_out rather than an input parameter!
    If the the length of wf_out plus t0_in extends past the end of wf_in,
    or if t0_in<0, pad the remaining values with NaN.
    
    Parameters
    ----------
    wf_in : array of float32/float64
            full waveform
    t0_in : float64
            start time of window. First index is floor(t0_in).
    wf_out: array of float32/float64
            windowed waveform
    """
    if len(wf_out) >= len(wf_in):
        raise DSPFatal("Windowed waveform must be smaller than input waveform!")

    if np.isnan(t0_in):
        wf_out[:] = np.nan
        return
    
    begin = min(np.floor(t0_in), len(wf_in))
    end = max(begin + len(wf_out), 0)
    if begin<0:
        wf_out[:len(wf_out)-end] = np.nan
        wf_out[len(wf_out)-end:] = wf_in[:end]
    elif end<len(wf_in):
        wf_out[:] = wf_in[begin:end]
    else:
        wf_out[:len(wf_in)-begin] = wf_in[begin:len(wf_in)]
        wf_out[len(wf_in)-begin:] = np.nan
