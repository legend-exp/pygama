import numpy as np
from numba import guvectorize
from .pole_zero import pole_zero, double_pole_zero
from iminuit import Minuit

class Model:
    """
    The model class containing the function to minimize.
    """
    errordef = Minuit.LEAST_SQUARES

    # the constructor
    def __init__(self, func, wf_in, baseline, beg, end):
        self.func = func
        self.x    = np.arange(beg, end)
        self.y    = np.asarray(wf_in, dtype=np.float64) - baseline
        self.beg  = beg
        self.end  = end

    # the function to minimize
    def __call__(self, args):
        y_pz = self.func(self.y, *args)[self.beg:self.end]
        return np.abs(np.sum(self.x) * np.sum(y_pz) - len(self.x) * np.sum(self.x * y_pz))

@guvectorize(["void(float32[:], uint16 , int32, int32, float32, float32[:])",
              "void(float64[:], uint16 , int64, int64, float64, float64[:])",
              "void(float32[:], float32, int32, int32, float32, float32[:])",
              "void(float64[:], float64, int64, int64, float64, float64[:])"],
             "(n),(),(),(),()->()", forceobj=True)
def optimize_1pz(wf_in, baseline, beg, end, p0, val0):
    """
    Find the optimal, single pole-zero cancellation's parameter
    by minimizing the slope in the waveform's specified time range.
    
    Parameters
    ----------
    wf_in   : array-like
              The input waveform
    baseline: int
              The resting baseline
    beg     : int
              The lower bound's index for the time range over
              which to optimize the pole-zero cancellation
    end     : int
              The upper bound's index for the time range over
              which to optimize the pole-zero cancellation
    p0      : int
              The initial guess of the optimal time constant
    val0    : int
              The output value of the best-fit time constant
              
    Processing Chain Example
    ------------------------
    "tau0": {
        "function": "optimize_1pz",
        "module": "pygama.dsp.processors",
        "args": ["waveform", "baseline", "0", "20*us", "500*us", "tau0"],
        "prereqs": ["waveform", "baseline"],
        "unit": "us"
    }
    """
    m = Minuit(Model(pole_zero, wf_in, baseline, beg, end), [p0])
    m.print_level = -1
    m.strategy = 1
    m.errordef = Minuit.LEAST_SQUARES
    m.migrad()
    val0[0] = m.values[0]
    
@guvectorize(["void(float32[:], uint16 , int32, int32, float32, float32, float32, float32[:], float32[:], float32[:])",
              "void(float64[:], uint16 , int64, int64, float64, float64, float64, float64[:], float64[:], float64[:])",
              "void(float32[:], float32, int32, int32, float32, float32, float32, float32[:], float32[:], float32[:])",
              "void(float64[:], float64, int64, int64, float64, float64, float64, float64[:], float64[:], float64[:])"],
             "(n),(),(),(),(),(),()->(),(),()", forceobj=True)
def optimize_2pz(wf_in, baseline, beg, end, p0, p1, p2, val0, val1, val2):
    """
    Find the optimal, double pole-zero cancellation's parameters
    by minimizing the slope in the waveform's specified time range.
    
    Parameters
    ----------
    wf_in   : array-like
              The input waveform
    baseline: int
              The resting baseline
    beg     : int
              The lower bound's index for the time range over
              which to optimize the pole-zero cancellation
    end     : int
              The upper bound's index for the time range over
              which to optimize the pole-zero cancellation
    p0      : int
              The initial guess of the optimal, longer time constant
    p1      : int
              The initial guess of the optimal, shorter time constant
    p2      : int
              The initial guess of the optimal fraction 
    val0    : int
              The output value of the best-fit, longer time constant
    val1    : int
              The output value of the best-fit, shorter time constant
    val2    : int
              The output value of the best-fit fraction
              
    Processing Chain Example
    ------------------------
    "tau1, tau2, frac": {
        "function": "optimize_2pz",
        "module": "pygama.dsp.processors",
        "args": ["waveform", "baseline", "0", "20*us", "500*us", "20*us", "0.02", "tau1", "tau2", "frac"],
        "prereqs": ["waveform", "baseline"],
        "unit": "us"
    }
    """
    m = Minuit(Model(double_pole_zero, wf_in, baseline, beg, end), [p0, p1, p2])
    m.print_level = -1
    m.strategy = 1
    m.errordef = Minuit.LEAST_SQUARES
    m.migrad()
    val0[0] = m.values[0]
    val1[0] = m.values[1]
    val2[0] = m.values[2]
