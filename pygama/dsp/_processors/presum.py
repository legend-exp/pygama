from numba import guvectorize

@guvectorize(["void(float32[:], float32[:])",
              "void(float64[:], float64[:])",
              "void(int32[:], int32[:])",
              "void(int64[:], int64[:])"],
             "(n),(m)", nopython=True, cache=True)
def presum(wf_in, wf_out):
    """Presum the waveform. Combine bins in chunks of len(wf_in)/len(wf_out),
    which is hopefully an integer. If it isn't, then some samples at the end
    will be omitted"""
    ps_fact = len(wf_in)//len(wf_out)
    for i in range(0, len(wf_out)):
        j0 = i*ps_fact
        wf_out[i] = wf_in[j0]
        for j in range(j0+1, j0+ps_fact):
            wf_out[i] += wf_in[j]
