from numba import guvectorize


@guvectorize(["void(float32[:], float32, float32[:])",
              "void(float64[:], float64, float64[:])"],
             "(n),()->(n)", nopython=True, cache=True)
def pole_zero(wf_in, tau, wf_out):
    """
    Pole-zero correction using time constant tau
    """
    const = np.exp(-1/tau)
    wf_out[0] = wf_in[0]
    for i in range(1, len(wf_in)):
        wf_out[i] = wf_out[i-1] + wf_in[i] - wf_in[i-1]*const
