import numpy as np
from numba import guvectorize
from math import log,exp

@guvectorize(["void(float32[:], float32, float32, float32, float32, float32[:])",
              "void(float64[:], float64, float64, float64, float64, float64[:])"],
              "(n),(), (), (), ()->(n)", nopython=True, cache=True)
def inject_sig_pulse(wf_in,t0,rise,amp,tau,wf_out):
    """
    Inject sigmoid pulse into existing waveform to simulate pileup
    """    
    wf_out[:] = wf_in[:]
    for T in range(len(wf_out)):
        wf_out[T] = wf_out[T]+ (amp/(1+exp((-1/rise)*(T-(t0+rise/2))))*exp(-(1/tau)*(T-t0)))

@guvectorize(["void(float32[:], float32, float32, float32, float32, float32[:])",
              "void(float64[:], float64, float64, float64, float64, float64[:])"],
              "(n),(), (), () , ()->(n)", nopython=True, cache=True)     
def inject_exp_pulse(wf_in,t0,rise,amp,tau,wf_out):
    """
    Inject exponential pulse into existing waveform to simulate pileup
    """
    wf_out[:] = wf_in[:]
    for T in range(len(wf_out)):
        if T < t0+rise:
            wf_out[T] = wf_out[T]+ ((exp(log(amp)*(T-t0-5)/(rise-5)))*exp(-(1/tau)*(T-t0)))
        else:
            wf_out[T] = wf_out[T]+ (amp*exp(-tau*(T-t0)))