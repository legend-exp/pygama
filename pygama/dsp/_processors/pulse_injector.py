import numpy as np
from numba import guvectorize
from math import log,exp

@guvectorize(["void(float32[:], float32, float32, float32, float32, float32[:])",
              "void(float64[:], float64, float64,float64,float64,float64[:])"],
              "(n),(),(),(), ()->(n)", nopython=True, cache=True)

def inject_sig_pulse(wf_in,t0,rt,A,decay, wf_out):
    """
    Inject sigmoid pulse into existing waveform to simulate pileup
    """


    wf_out[:] = np.nan

    if np.isnan(wf_in).any() or np.isnan(rt) or np.isnan(t0) or np.isnan(A) or np.isnan(decay):
        return

    rise = 4*log(99)/rt

    wf_out[:] = wf_in[:]
    for T in range(len(wf_out)):
        wf_out[T] = wf_out[T]+ (A/(1+exp((-rise)*(T-(t0+rt/2))))*exp(-(1/decay)*(T-t0)))

@guvectorize(["void(float32[:], float32, float32, float32, float32, float32[:])",
              "void(float64[:], float64, float64,float64,float64,float64[:])"],
              "(n),(),(),(), ()->(n)", nopython=True, cache=True)

def inject_exp_pulse(wf_in,t0,rt,A,decay, wf_out):
    """
    Inject exponential pulse into existing waveform to simulate pileup
    """

    wf_out[:] = np.nan

    if np.isnan(wf_in).any() or np.isnan(rt) or np.isnan(t0) or np.isnan(A) or np.isnan(decay):
        return

    wf_out[:] = wf_in[:]
    for T in range(len(wf_out)):
        if (t0<= T)& (T<= t0+rt):
            wf_out[T] += ((A*exp((T-t0-rt)/(rt))*exp(-(1/decay)*(T-t0))))
        elif (T>t0+rt):
            wf_out[T] += (A*exp(-(1/decay)*(T-t0)))
