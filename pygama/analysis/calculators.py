import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d
from scipy import signal, interpolate

#Finds the maximum current ("A").  Current is calculated by convolution with a first-deriv. of Gaussian
def current_max(waveform, sigma=1):
  if sigma > 0:
      return np.amax(gaussian_filter1d(waveform, sigma=sigma, order=1))
  else:
      print("Current max requires smooth>0")
      exit(0)

#Finds baseline from start index to end index samples (default linear)
def fit_baseline(waveform, start_index=0, end_index=500, order=1):
    if end_index == -1: end_index = len(waveform)
    p = np.polyfit(np.arange(start_index, end_index), waveform[start_index:end_index], 1)
    return p

def is_saturated(waveform, bit_precision=14):
    return True if np.amax(waveform) >= 0.5*2**bit_precision - 1 else False

#Estimate t0
def t0_estimate(waveform, baseline=0, median_kernel_size=51, max_t0_adc=100):
    '''
    max t0 adc: maximum adc (above baseline) the wf can get to before assuming the wf has started
    '''

    if np.amax(waveform)<max_t0_adc:
        return np.nan

    wf_med = signal.medfilt(waveform, kernel_size=median_kernel_size)
    med_diff = gaussian_filter1d(wf_med, sigma=1, order=1)

    tp05 = calc_timepoint(waveform, percentage=max_t0_adc, baseline=0, do_interp=False, doNorm=False)
    tp05_rel = np.int(tp05+1)
    thresh = 5E-5
    last_under = tp05_rel - np.argmax(med_diff[tp05_rel::-1]<=thresh)
    if last_under >= len(med_diff)-1:
        last_under = len(med_diff)-2

    t0 = np.interp(thresh, ( med_diff[last_under],   med_diff[last_under+1] ), (last_under, last_under+1))

    return t0

#Estimate t0
def max_time(waveform):
    return np.argmax(waveform)

#Estimate arbitrary timepoint before max
def calc_timepoint(waveform, percentage=0.5, baseline=0, do_interp=False, doNorm=True, norm=None):
    '''
    percentage: if less than zero, will return timepoint on falling edge
    do_interp: linear linerpolation of the timepoint...
    '''
    wf_norm = (np.copy(waveform) - baseline)

    if doNorm:
        if norm is None: norm = np.amax(wf_norm)
        wf_norm /= norm

    def get_tp(perc):
        if perc > 0:
            first_over = np.argmax( wf_norm >= perc )
            if do_interp and first_over > 0:
                val = np.interp(perc, ( wf_norm[first_over-1],   wf_norm[first_over] ), (first_over-1, first_over))
            else: val = first_over
        else:
            perc = np.abs(perc)
            above_thresh = wf_norm >= perc
            last_over = len(wf_norm)-1 - np.argmax(above_thresh[::-1])
            if do_interp and last_over < len(wf_norm)-1:
                val = np.interp(perc, (  wf_norm[last_over+1], wf_norm[last_over] ), (last_over+1, last_over))
            else: val = last_over
        return val

    if not getattr(percentage, '__iter__', False):
        return get_tp(percentage)
    else:
        vfunc = np.vectorize(get_tp)
        return vfunc(percentage)


#Calculate maximum of trapezoid -- no pride here
def trap_max(waveform, method = "max", pickoff_sample = 0):
    if method == "max": return np.amax(waveform)
    elif method == "fixed_time": return waveform[pickoff_sample]
