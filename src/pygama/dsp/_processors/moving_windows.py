import numpy as np
from numba import guvectorize

from pygama.dsp.errors import DSPFatal


@guvectorize(["void(float32[:], float32, float32[:])",
              "void(float64[:], float64, float64[:])"],
             "(n),()->(n)", nopython=True, cache=True)
def moving_window_left(w_in, length, w_out):
    """
    Apply a moving-average window to the waveform from the left.

    Parameters
    ----------
    w_in  : array-like
            The input waveform
    length: int
            The length of the moving window
    w_out : array-like
            The windowed waveform

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
    """
    w_out[:] = np.nan

    if np.isnan(w_in).any():
        return

    if np.floor(length) != length:
        raise DSPFatal('The length of the moving window must be an integer')

    if int(length) < 0 or int(length) >= len(w_in):
        raise DSPFatal('The length of the moving window is out of range')

    w_out[0] = w_in[0] / length
    for i in range(1, int(length)):
        w_out[i] = w_out[i-1] + w_in[i] / length
    for i in range(int(length), len(w_in)):
        w_out[i] = w_out[i-1] + (w_in[i] - w_in[i-int(length)]) / length

@guvectorize(["void(float32[:], float32, float32[:])",
              "void(float64[:], float64, float64[:])"],
             "(n),()->(n)", nopython=True, cache=True)
def moving_window_right(w_in, length, w_out):
    """
    Apply a moving-average window to the waveform from the left.

    Parameters
    ----------
    w_in  : array-like
            The input waveform
    length: int
            The length of the moving window
    w_out : array-like
            The windowed waveform

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
    """
    w_out[:] = np.nan

    if np.isnan(w_in).any():
        return

    if np.floor(length) != length:
        raise DSPFatal('The length of the moving window must be an integer')

    if int(length) < 0 or int(length) >= len(w_in):
        raise DSPFatal('The length of the moving window is out of range')

    w_out[-1] = w_in[-1]
    for i in range(len(w_in) - 2, len(w_in) - int(length) - 1, -1):
        w_out[i] = w_out[i+1] + (w_in[i] - w_out[-1]) / length
    for i in range(len(w_in) - int(length) - 1, -1, -1):
        w_out[i] = w_out[i+1] + (w_in[i] - w_in[i+int(length)]) / length

@guvectorize(["void(float32[:], float32, float32, float32[:])",
              "void(float64[:], float64, float64, float64[:])"],
             "(n),(),()->(n)", nopython=True, cache=True)
def moving_window_multi(w_in, length, num_mw, w_out):
    """
    Apply a series of moving-average windows to the waveform, alternating
    its application between the left and the right.

    Parameters
    ----------
    w_in  : array-like
            The input waveform
    length: int
            The length of the moving window
    num_mw: int
            The number of moving windows
    w_out : array-like
            The windowed waveform

    Examples
    --------
    .. code-block :: json

        "curr_av": {
            "function": "moving_window_multi",
            "module": "pygama.dsp.processors",
            "args": ["curr", "96*ns", "3", "curr_av"],
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

    wf_buf = w_in.copy()
    for i in range(0, int(num_mw), 1):
        if i % 2 == 1:
            w_out[-1] = w_in[-1]
            for i in range(len(w_in) - 2, len(w_in) - int(length) - 1, -1):
                w_out[i] = w_out[i+1] + (w_in[i] - w_out[-1]) / length
            for i in range(len(wf_buf) - int(length) - 1, -1, -1):
                w_out[i] = w_out[i+1] + (wf_buf[i] - wf_buf[i+int(length)]) / length
        else:
            w_out[0] = wf_buf[0] / length
            for i in range(1, int(length), 1):
                w_out[i] = w_out[i-1] + wf_buf[i] / length
            for i in range(int(length), len(w_in), 1):
                w_out[i] = w_out[i-1] + (wf_buf[i] - wf_buf[i-int(length)]) / length
        wf_buf[:] = w_out[:]

@guvectorize(["void(float32[:], float32, float32[:])",
              "void(float64[:], float64, float64[:])"],
             "(n),(),(m)", nopython=True, cache=True)
def avg_current(w_in, length, w_out):
    """
    Calculate the derivative of the waveform, averaged across the specified
    number of samples.

    Parameters
    ----------
    w_in  : array-like
            The input waveform
    length: int
            The length of the moving window
    w_out : array-like
            The output derivative

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

    if np.isnan(w_in).any():
        return

    if np.floor(length) != length:
        raise DSPFatal('The length of the moving window must be an integer')

    if int(length) < 0 or int(length) >= len(w_in):
        raise DSPFatal('The length of the moving window is out of range')

    w_out[:] = w_in[int(length):] - w_in[:-int(length)]
    w_out /= length
