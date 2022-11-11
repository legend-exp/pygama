from __future__ import annotations

import numpy as np
from numba import guvectorize
import scipy.signal as signal

import pygama.lgdo.lh5_store as lh5
from pygama.dsp.errors import DSPFatal
from pygama.dsp.utils import numba_defaults_kwargs as nb_kwargs

#def dplms_filter(file_name_array: list[str]) -> np.ndarray:

def dplms_filter(baselines: np.ndarray, reference: np.ndarray, length: int, a1: float, a2: int, a3: int, ff: int, diff: bool) -> Callable:
    
    # noise matrix
    if diff: baselines = np.diff(baselines)
    nwf = int(baselines.shape[0])
    bsize = baselines.shape[1]
    nmat =  np.matmul(baselines.transpose(), baselines)/nwf
    nmat = signal.convolve2d(nmat, np.identity(bsize-length+1),
                            boundary='symm', mode='valid')/(bsize-length+1)
    
    # reference matrix
    ssize = len(reference)
    flo = int(ssize/2 - length/2)
    fhi = int(ssize/2 + length/2)
    rmat = np.zeros([length,length])
    rsig = np.zeros([length])
    if ff == 0: ff = [0]
    else: ff = [-1,0,1]
    for i in ff:
        rmat += np.outer(reference[flo+i:fhi+i], reference[flo+i:fhi+i])        
        rsig +=  reference[flo+i:fhi+i]
    rmat /= len(ff)
    rsig = np.transpose(rsig)/len(ff)
    
    # filter calculation
    mat = a1*nmat + a2*rmat + a3*np.ones([length,length])
    x = np.linalg.solve(mat, rsig)
    conv = signal.convolve(reference, np.flip(x), mode = 'valid')
    
    @guvectorize(
        ["void(float32[:], float32[:])", "void(float64[:], float64[:])"],
        "(n),(m)",
        **nb_kwargs(
            cache=False,
            forceobj=True,
        ),
    )
    
    def dplms_out(w_in: np.ndarray, w_out: np.ndarray) -> None:
        """
        Parameters
        ----------
        w_in
            the input waveform.
        w_out
            the filtered waveform.
        """
        
        w_out[:] = np.nan

        if np.isnan(w_in).any():
            return

        if len(x) > len(w_in):
            raise DSPFatal("The filter is longer than the input waveform")

        w_out[:] = np.convolve(w_in, x, "valid")
        
    return dplms_out


