import numpy as np
from iminuit import Minuit
from numba import guvectorize

from pygama.dsp.errors import DSPFatal

from .pole_zero import double_pole_zero, pole_zero


class Model:
    """
    The model class containing the function to minimize.
    """
    errordef = Minuit.LEAST_SQUARES

    # the constructor
    def __init__(self, func, w_in, baseline, beg, end):
        self.func = func
        self.x    = np.arange(beg, end)
        self.y    = np.asarray(w_in, dtype=np.float64) - baseline
        self.beg  = beg
        self.end  = end

    # the function to minimize
    def __call__(self, args):
        y_pz = self.func(self.y, *args)[self.beg:self.end]
        return np.abs(np.sum(self.x) * np.sum(y_pz) - len(self.x) * np.sum(self.x * y_pz))

@guvectorize(["void(float32[:], float32, float32, float32, float32, float32[:])",
              "void(float64[:], float64, float64, float64, float64, float64[:])"],
             "(n),(),(),(),()->()", forceobj=True)
def optimize_1pz(w_in, a_baseline_in, t_beg_in, t_end_in, p0_in, val0_out):
    """
    Find the optimal, single pole-zero cancellation's parameter
    by minimizing the slope in the waveform's specified time range.

    Parameters
    ----------
    w_in         : array-like
                   The input waveform
    a_baseline_in: float
                   The resting baseline
    t_beg_in     : int
                   The lower bound's index for the time range over
                   which to optimize the pole-zero cancellation
    t_end_in     : int
                   The upper bound's index for the time range over
                   which to optimize the pole-zero cancellation
    p0_in        : float
                   The initial guess of the optimal time constant
    val0_out     : float
                   The output value of the best-fit time constant

    Examples
    --------
    .. code-block :: json

        "tau0": {
            "function": "optimize_1pz",
            "module": "pygama.dsp.processors",
            "args": ["waveform", "baseline", "0", "20*us", "500*us", "tau0"],
            "prereqs": ["waveform", "baseline"],
            "unit": "us"
        }
    """
    val0_out[0] = np.nan

    if np.isnan(w_in).any() or np.isnan(a_baseline_in) or np.isnan(t_beg_in) or np.isnan(t_end_in) or\
       np.isnan(p0_in):
        return

    if not np.floor(t_beg_in) == t_beg_in or\
       not np.floor(t_end_in) == t_end_in:
        raise DSPFatal('The waveform index is not an integer')

    if int(t_beg_in) < 0 or int(t_beg_in) > len(w_in) or\
       int(t_end_in) < 0 or int(t_end_in) > len(w_in):
        raise DSPFatal('The waveform index is out of range')

    m = Minuit(Model(pole_zero, w_in, a_baseline_in, int(t_beg_in), int(t_end_in)), [p0_in])
    m.print_level = -1
    m.strategy = 1
    m.errordef = Minuit.LEAST_SQUARES
    m.migrad()
    val0_out[0] = m.values[0]

@guvectorize(["void(float32[:], float32, float32, float32, float32, float32, float32, float32[:], float32[:], float32[:])",
              "void(float64[:], float64, float64, float64, float64, float64, float64, float64[:], float64[:], float64[:])"],
             "(n),(),(),(),(),(),()->(),(),()", forceobj=True)
def optimize_2pz(w_in, a_baseline_in, t_beg_in, t_end_in, p0_in, p1_in, p2_in, val0_out, val1_out, val2_out):
    """
    Find the optimal, double pole-zero cancellation's parameters
    by minimizing the slope in the waveform's specified time range.

    Parameters
    ----------
    w_in         : array-like
                   The input waveform
    a_baseline_in: float
                   The resting baseline
    t_beg_in     : int
                   The lower bound's index for the time range over
                   which to optimize the pole-zero cancellation
    t_end_in     : int
                   The upper bound's index for the time range over
                   which to optimize the pole-zero cancellation
    p0_in        : float
                   The initial guess of the optimal, longer time constant
    p1_in        : float
                   The initial guess of the optimal, shorter time constant
    p2_in        : float
                   The initial guess of the optimal fraction
    val0_out     : float
                   The output value of the best-fit, longer time constant
    val1_out     : float
                   The output value of the best-fit, shorter time constant
    val2_out     : float
                   The output value of the best-fit fraction

    Examples
    --------
    .. code-block :: json

        "tau1, tau2, frac": {
            "function": "optimize_2pz",
            "module": "pygama.dsp.processors",
            "args": ["waveform", "baseline", "0", "20*us", "500*us", "20*us", "0.02", "tau1", "tau2", "frac"],
            "prereqs": ["waveform", "baseline"],
            "unit": "us"
        }
    """
    val0_out[0] = np.nan
    val1_out[0] = np.nan
    val2_out[0] = np.nan

    if np.isnan(w_in).any() or np.isnan(a_baseline_in) or np.isnan(t_beg_in) or np.isnan(t_end_in) or\
       np.isnan(p0_in) or np.isnan(p1_in) or np.isnan(p2_in):
        return

    if not np.floor(t_beg_in) == t_beg_in or\
       not np.floor(t_end_in) == t_end_in:
        raise DSPFatal('The waveform index is not an integer')

    if int(t_beg_in) < 0 or int(t_beg_in) > len(w_in) or\
       int(t_end_in) < 0 or int(t_end_in) > len(w_in):
        raise DSPFatal('The waveform index is out of range')

    m = Minuit(Model(double_pole_zero, w_in, a_baseline_in, int(t_beg_in), int(t_end_in)), [p0_in, p1_in, p2_in])
    m.print_level = -1
    m.strategy = 1
    m.errordef = Minuit.LEAST_SQUARES
    m.migrad()
    val0_out[0] = m.values[0]
    val1_out[0] = m.values[1]
    val2_out[0] = m.values[2]
