import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
def blsub(wf):
    return wf - np.mean(wf[0:25])
    
def pz(wf, decay, clk=100e6):
    """
    pole-zero correct a waveform
    decay is in us, clk is in Hz
    """
    # get linear filter parameters, in units of [clock ticks]
    dt = decay * (1e10 / clk)
    rc = 1 / np.exp(1 / dt)
    num, den = [1, -1], [1, -rc]

    # reversing num and den does the inverse transform (ie, PZ corrects)
    return signal.lfilter(den, num, wf)

def trap(waveform, rampTime=150, flatTime=250, decayTime=0, baseline=0.):
    """
    Apply a trap filter to a waveform.
    """
    decayConstant = 0.
    norm = rampTime
    if decayTime != 0:
        decayConstant = 1. / (np.exp(1. / decayTime) - 1)
        norm *= decayConstant

    trapOutput = np.zeros_like(waveform)
    fVector = np.zeros_like(waveform)
    scratch = np.zeros_like(waveform)
    fVector[0] = waveform[0] - baseline
    trapOutput[0] = (decayConstant + 1.) * (waveform[0] - baseline)

    wf_minus_ramp = np.zeros_like(waveform)
    wf_minus_ramp[:rampTime] = baseline
    wf_minus_ramp[rampTime:] = waveform[:len(waveform) - rampTime]

    wf_minus_ft_and_ramp = np.zeros_like(waveform)
    wf_minus_ft_and_ramp[:(flatTime + rampTime)] = baseline
    wf_minus_ft_and_ramp[(flatTime + rampTime):] = waveform[:len(waveform) - flatTime - rampTime]

    wf_minus_ft_and_2ramp = np.zeros_like(waveform)
    wf_minus_ft_and_2ramp[:(flatTime + 2 * rampTime)] = baseline
    wf_minus_ft_and_2ramp[(flatTime + 2 * rampTime):] = waveform[:len(waveform) - flatTime -
                                              2 * rampTime]

    scratch = waveform - (wf_minus_ramp +
                          wf_minus_ft_and_ramp - # NOTE: clint changed this to - after walter convinced him
                          wf_minus_ft_and_2ramp)

    if decayConstant != 0:
        fVector = np.cumsum(fVector + scratch)
        trapOutput = np.cumsum(trapOutput + fVector + decayConstant * scratch)
    else:
        trapOutput = np.cumsum(trapOutput + scratch)

    # Normalize and resize output
    tmp_hi = len(waveform) - (2 * rampTime + flatTime)
    trapOutput[:tmp_hi] = trapOutput[2 * rampTime + flatTime:] / norm
    trapOutput.resize((len(waveform) - (2 * rampTime + flatTime)))
    return trapOutput

def current(wf, sigma=5):
    """
    calculate the current trace,
    by convolving w/ first derivative of a gaussian.
    """

    wfc = ndimage.filters.gaussian_filter1d(wf, sigma=sigma, order=1) # lol

    return wfc

