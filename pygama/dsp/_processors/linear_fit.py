from numba import guvectorize

@guvectorize(["void(float32[:], float32[:], float32[:])",
              "void(float64[:], float64[:], float64[:])",
              "void(int32[:], float32[:], float32[:])",
              "void(int64[:], float64[:], float64[:])"],
             "(n)->(),()", nopython=True, cache=True)
def linear_fit(wf, b, m):
    """
    Perform a linear fit of the waveform to m*x + b. Followed
    the numerical recipes in C (ch 15, p 664) algorithm
    """
    S = len(wf)
    Sx = S*(S-1)/2.
    Stt = 0.
    Sty = 0.
    Sy = 0.
    for i, samp in enumerate(wf):
        t = i - Sx/S
        Stt += t*t
        Sty += t*samp
        Sy += samp
    m[0] = Sty/Stt
    b[0] = (Sy-Sx*m[0])/S
