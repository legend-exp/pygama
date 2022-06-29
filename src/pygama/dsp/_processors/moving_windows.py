import numpy as np
from numba import guvectorize

from pygama.dsp.errors import DSPFatal


@guvectorize(["void(float32[:], float32, float32[:])",
              "void(float64[:], float64, float64[:])"],
             "(n),()->(n)", nopython=True, cache=True)
def moving_window_left(w_in, length, w_out):
    '''
    Applies a moving average window to the waveform from the left, assumes that the baseline is at 0.

    Parameters
    ----------
    w_in : array-like
        The input waveform
    length : float
        Length of the moving window to be applied
    w_out : array-like
        Output waveform after moving window applied

    Examples
    --------
    .. code-block :: json

        "wf_mw": {
            "function": "moving_window_left",
            "module": "pygama.dsp.processors",
            "args": ["wf_pz", "96*ns", "wf_mw"],
            "unit": "ADC",
            "prereqs": ["wf_pz"]
        }
    '''

    w_out[:] = np.nan

    if (np.isnan(w_in).any()):
        return

    if (not length >= 0 or not length< len(w_in)):
        raise DSPFatal('length is out of range, must be between 0 and the length of the waveform')

    w_out[0] = w_in[0]
    for i in range(1, int(length)):
        w_out[i] = w_out[i-1] + (w_in[i] - w_in[0]) / length
    for i in range(int(length), len(w_in)):
        w_out[i] = w_out[i-1] + (w_in[i] - w_in[i-int(length)]) / length


@guvectorize(["void(float32[:], float32, float32[:])",
              "void(float64[:], float64, float64[:])"],
             "(n),()->(n)", nopython=True, cache=True)
def moving_window_right(w_in, length, w_out):
    '''
    Applies a moving average window to the waveform from the right.

    Parameters
    ----------
    w_in : array-like
        The input waveform
    length : float
        Length of the moving window to be applied
    w_out : array-like
        Output waveform after moving window applied

    Examples
    --------
    .. code-block :: json

        "wf_mw": {
            "function": "moving_window_right",
            "module": "pygama.dsp.processors",
            "args": ["wf_pz", "96*ns", "wf_mw"],
            "unit": "ADC",
            "prereqs": ["wf_pz"]
        }
    '''

    w_out[:] = np.nan

    if (np.isnan(w_in).any()):
        return

    if (not length >= 0 or not length< len(w_in)):
        raise DSPFatal('length is out of range, must be between 0 and the length of the waveform')

    w_out[-1]= w_in[-1]
    for i in range(1, int(length),1):
        w_out[len(w_in)-1-i] = w_out[len(w_in)-i] + (w_in[len(w_in)-1-i]-w_out[-1])/length
    for i in range(int(length), len(w_in), 1):
        w_out[len(w_in)-1-i] = w_out[len(w_in)-i] + (w_in[len(w_in)-1-i] - w_in[len(w_in)-1-i+int(length)])/length


@guvectorize(["void(float32[:], float32, float32, int32, float32[:])",
              "void(float64[:], float64, float64, int32, float64[:])"],
             "(n),(),(),()->(n)", nopython=True, cache=True)
def moving_window_multi(w_in, length, num_mw, mw_type,w_out):
    """
    Apply a series of moving-average windows to the waveform, alternating
    its application between the left and the right.
    Parameters
    ----------
    w_in : array-like
        The input waveform
    length : float
        Length of the moving window to be applied
    num_mw : int
        The number of moving windows
    mw_type: int
        0: alternate moving windows right and left
        1: only left
        2: only right
    w_out : array-like
        The windowed waveform

    Examples
    --------
    .. code-block :: json

        "curr_av": {
            "function": "moving_window_multi",
            "module": "pygama.dsp.processors",
            "args": ["curr", "96*ns", "3", "0", "curr_av"],
            "unit": "ADC/sample",
            "prereqs": ["curr"]
        }
    """
    w_out[:] = np.nan

    if np.isnan(w_in).any():
        return

    if np.floor(length) != length:
        raise DSPFatal('The length of the moving window must be an integer')

    if np.floor(num_mw) != num_mw:
        raise DSPFatal('The number of moving windows must be an integer')

    if int(length) < 0 or int(length) >= len(w_in):
        raise DSPFatal('The length of the moving window is out of range')

    if int(num_mw) < 0:
        raise DSPFatal('The number of moving windows much be positive')

    w_buf = w_in.copy()
    for i in range(0, int(num_mw), 1):
        if ((i % 2 == 1) & (mw_type==0))|(mw_type==2):
            w_out[-1]= w_buf[-1]
            for i in range(1, int(length),1):
                w_out[len(w_buf)-1-i] = w_out[len(w_buf)-i] + (w_buf[len(w_buf)-1-i]-w_out[-1])/length
            for i in range(int(length), len(w_buf), 1):
                w_out[len(w_buf)-1-i] = w_out[len(w_buf)-i] + (w_buf[len(w_buf)-1-i] - w_buf[len(w_buf)-1-i+int(length)])/length
        else:
            w_out[0] = w_buf[0]
            for i in range(1, int(length)):
                w_out[i] = w_out[i-1] + (w_buf[i] - w_buf[0]) / length
            for i in range(int(length), len(w_buf)):
                w_out[i] = w_out[i-1] + (w_buf[i] - w_buf[i-int(length)]) / length
        w_buf = w_out.copy()




@guvectorize(["void(float32[:], float32, float32[:])",
              "void(float64[:], float64, float64[:])"],
             "(n),(),(m)", nopython=True, cache=True)
def avg_current(w_in, length, w_out):
    """
    Calculate the derivative of a WF, averaged across n samples. Dimension of
    deriv should be len(wf) - n

    Parameters
    ----------
    w_in : array-like
        The input waveform
    length : float
        Length of the moving window to be applied
    w_out : array-like
        Output waveform after derivation

    Examples
    --------
    .. code-block :: json

        "curr": {
            "function": "avg_current",
            "module": "pygama.dsp.processors",
            "args": ["wf_pz", 1, "curr(len(wf_pz)-1, f)"],
            "unit": "ADC/sample",
            "prereqs": ["wf_pz"]
        }
    """

    w_out[:] = np.nan

    if (np.isnan(w_in).any()):
        return

    if (not length >= 0 or not length< len(w_in)):
        raise DSPFatal('length is out of range, must be between 0 and the length of the waveform')


    w_out[:] = w_in[int(length):] - w_in[:-int(length)]
    w_out/=length
