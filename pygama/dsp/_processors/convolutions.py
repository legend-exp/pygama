import numpy as np
from numba import guvectorize
import math
from math import pow
from pygama.dsp.errors import DSPFatal

def cusp_filter(length, sigma, flat, decay):
    
    """
    This processor applies a CUSP filter to the waveform for use in energy estimation. 
    It is composed of a factory function that is called using the init_args argument and the function the waveforms are passed to using args
    Initialisation Parameters
    ------------------------
    length : int 
             The length of the filter to be convolved with the waveforms generally specified as len(wf)-some number
    sigma : int, float
             Parameter determining the curvature of the rising and faling part of the kernel
    flat : int
            Size of the flat section of the CUSP filter 
    decay : int
            Decay constant of the exponential to be convolved with the CUSP kernel
    Parameters
    ----------
    w_in : array-like
            waveform to be convolved with CUSP filter
    w_out : array-like
        waveform convolved with CUSP filter
    Processing Chain Example
    ------------------------
    "wf_cusp": {
        "function": "cusp_filter",
        "module": "pygama.dsp.processors",
        "args": [ "wf_blsub", "wf_cusp(101,f)" ],
        "init_args" : ["len(wf_blsub)-100", "40*us", "3*us", "45*us"],
        "prereqs": [ "wf_blsub" ],
        "unit": "ADC"
        },
    """

    if (not length > 0):
        raise DSPFatal('length out of range, must be greater than 0')
    if (not sigma >= 0):
        raise DSPFatal('sigma out of range, must be >= 0')

    if (not flat >= 0):
        raise DSPFatal('flat out of range, must be >= 0')
    if (not decay >= 0):
        raise DSPFatal('decay out of range, must be >= 0')

    lt = int((length-flat)/2)
    flat_int = int(flat)
    cusp = np.zeros(length)
    for ind in range(lt):
        cusp[ind] = float(math.sinh(ind/sigma)/math.sinh(lt/sigma))
    for ind in range(lt, lt+flat_int+1, 1):
        cusp[ind] = 1
    for ind in range(lt+flat_int+1, length,1):
        cusp[ind] = float(math.sinh((length-ind)/sigma)/math.sinh(lt/sigma))


    den = [1, -np.exp(-1/decay)]
    cuspd = np.convolve(cusp, den, 'same')
    
    @guvectorize(["void(float32[:], float32[:])",
                  "void(float64[:], float64[:])"],
                 "(n),(m)", forceobj=True)
    def cusp_out(w_in,w_out):

        w_out[:]= np.nan

        if (np.isnan(w_in).any()):
            return

        if (len(cuspd)> len(w_in)):
            raise DSPFatal('Filter longer than input waveform')

        w_out[:] = np.convolve(w_in, cuspd, 'valid')
    return cusp_out

def zac_filter(length, sigma, flat, decay):
    
    """
    This processor applies a ZAC (Zero Area Cusp) filter to the waveform for use in energy estimation. 
    It is composed of a factory function that is called using the init_args argument and the function the waveforms are passed to using args
    Initialisation Parameters
    ------------------------
    length : int 
             The length of the filter to be convolved with the waveforms generally specified as len(wf)-some number
    sigma : int, float
             Parameter determining the curvature of the rising and faling part of the kernel
    flat : int
            Size of the flat section of the ZAC filter 
    decay : int
            Decay constant of the exponential to be convolved with the ZAC kernel
    Parameters
    ----------
    w_in : array-like
            waveform to be convolved with ZAC filter
    w_out : array-like
        waveform convolved with ZAC filter
    Processing Chain Example
    ------------------------
    "wf_zac": {
        "function": "zac_filter",
        "module": "pygama.dsp.processors",
        "args": [ "wf_blsub", "wf_zac(101,f)" ],
        "init_args" : ["len(wf_blsub)-100", "40*us", "3*us", "45*us"],
        "prereqs": [ "wf_blsub" ],
        "unit": "ADC"
        },
    """

    if (not length > 0):
        raise DSPFatal('length out of range, must be greater than 0')
    if (not sigma >= 0):
        raise DSPFatal('sigma out of range, must be >= 0')

    if (not flat >= 0):
        raise DSPFatal('flat out of range, must be >= 0')
    if (not decay >= 0):
        raise DSPFatal('decay out of range, must be >= 0')

    lt = int((length-flat)/2)
    flat_int = int(flat)
    # calculate cusp filter and negative parables
    cusp = np.zeros(length)
    par = np.zeros(length)
    for ind in range(lt):
        cusp[ind] = float(math.sinh(ind/sigma)/math.sinh(lt/sigma))
        par[ind] = pow(ind-lt/2,2)-pow(lt/2,2)
    for ind in range(lt, lt+flat_int+1, 1):
        cusp[ind] = 1
    for ind in range(lt+flat_int+1, length,1):
        cusp[ind] = float(math.sinh((length-ind)/sigma)/math.sinh(lt/sigma))
        par[ind] = pow(length-ind-lt/2,2)-pow(lt/2,2)

    # calculate area of cusp and parables
    areapar, areacusp = 0, 0
    for i in range(length):
        areapar += par[i]
        areacusp += cusp[i]
    #normalize parables area
    par = -par/areapar*areacusp
    #create zac filter
    zac = cusp + par
    #deconvolve zac filter
    den = [1, -np.exp(-1/decay)]
    zacd = np.convolve(zac, den, 'same')

    @guvectorize(["void(float32[:], float32[:])",
                  "void(float64[:], float64[:])"],
                 "(n),(m)", forceobj=True)
    def zac_out(w_in,w_out):

        w_out[:] = np.nan

        if (np.isnan(w_in).any()):
            return

        if (len(zacd) > len(w_in)):
            raise DSPFatal('Filter longer than input waveform')

        w_out[:] = np.convolve(w_in, zacd, 'valid')
    return zac_out

def t0_filter(rise,fall):

    """
    This processor applies modified assymetric trap filter to the waveform for use in t0 estimation. 
    It is composed of a factory function that is called using the init_args argument and the function the waveforms are passed to using args.
    Initialisation Parameters
    ------------------------
    rise : int 
            Specifies length of rise section. This is the linearly increasing section of the filter that performs a weighted average.
    fall : int
            Specifies length of fall section. This is the simple averaging part of the filter.
    Parameters
    ----------
    w_in : array-like
            waveform to be convolved with t0 filter
    w_out : array-like
        waveform convolved with t0 filter
    Processing Chain Example
    ------------------------
    "wf_t0_filter": {
        "function": "t0_filter",
        "module": "pygama.dsp.processors",
        "args": [ "wf_pz", "wf_t0_filter(3748,f)" ],
        "init_args" : ["128*ns", "2*us"],
        "prereqs": ["wf_pz"],
        "unit": "ADC"
        },
    """

    if (not rise >= 0):
        raise DSPFatal('rise out of range, must be >= 0')
    if (not fall >= 0):
        raise DSPFatal('fall out of range, must be >= 0')

    t0_kern = np.arange(2/float(rise),0, -2/(float(rise)**2))
    t0_kern = np.append(t0_kern, np.zeros(int(fall))-(1/float(fall)))

    @guvectorize(["void(float32[:], float32[:])",
                  "void(float64[:], float64[:])"],
                 "(n),(m)", forceobj=True)
    def t0_filter_out(w_in,w_out):

        w_out[:] = np.nan

        if (np.isnan(w_in).any()):
            return

        if (len(t0_kern)> len(w_in)):
            raise DSPFatal('Filter longer than input waveform')

        w_out[:] = np.convolve(w_in, t0_kern)[:len(w_in)]
    return t0_filter_out
