import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import lfilter

def bl_subtract(waves, calcs, test=False):
    """
    Return an ndarray of baseline-subtracted waveforms
    Depends on fit_baseline calculator.
    for reference, the non-vector version is just:
    return waveform - (bl_0 + bl_1 * np.arange(len(waveform)))
    """
    wfs = waves["waveform"]
    nwfs, nsamp = wfs.shape[0], wfs.shape[1]

    bl_0 = calcs["bl_int"].values[:,np.newaxis]

    slope_vals = calcs["bl_slope"].values[:,np.newaxis]
    bl_1 = np.tile(np.arange(nsamp), (nwfs, 1)) * slope_vals

    # blsub_wfs = wfs - bl_0
    blsub_wfs = wfs - (bl_0 + bl_1)

    if test:
        # alternate - transform based off avg_baseline calcsulator
        # bl_avg = calcs["bl_avg"].values[:,np.newaxis]
        # blsub_avgs = wfs - bl_avg

        iwf = 1
        plt.plot(np.arange(nsamp), wfs[iwf], '-g', label="raw")
        plt.plot(np.arange(nsamp), blsub_wfs[iwf], '-b', label="bl_sub")
        # plt.plot(np.arange(nsamp), blsub_avgs[iwf], '-g', label="bl_avg")
        plt.legend()
        plt.show()
        exit()

    return {"wf_blsub": blsub_wfs} # note, floats are gonna take up more memory


def trap_filter(waves, calcs, rt=400, ft=200, decay_time=0, test=False):
    """
    vectorized trapezoid filter
    """
    wfs = waves["wf_blsub"]
    nwfs, nsamp = wfs.shape[0], wfs.shape[1]

    wfs_minus_ramp = np.zeros_like(wfs) + 0 # baseline
    wfs_minus_ramp[:, rt:] = wfs[:, :nsamp - rt]

    wfs_minus_ft_and_ramp = np.zeros_like(wfs)
    wfs_minus_ft_and_ramp[:, (ft + rt):] = wfs[:, :nsamp - ft - rt]

    wfs_minus_trap = np.zeros_like(wfs)
    wfs_minus_trap[:, (ft + 2*rt):] = wfs[:, :nsamp - ft - 2*rt]

    trap_wfs = np.zeros_like(wfs)
    scratch = wfs - wfs_minus_ramp - wfs_minus_ft_and_ramp + wfs_minus_trap

    # # apply pole-zero correction
    # if decay_time != 0:
    #     decay_const = 1. / (np.exp(1. / decay_time - 1))
    #     pz_corr = np.zeros_like(wfs)
    #     pz_corr[:,0] = wfs[:,0]
    #     pz_wfs = np.cumsum(trap_wfs + pz_corr + decay_const * scratch,
    #                          axis=1) / (rt * decay_const)

    trap_wfs = np.cumsum(scratch, axis=1) / rt

    if test:
        import pygama.dsp.transforms as pt # maybe try importing ben's directly
        iwf = 5

        # see wtf the algorithm is actually doing
        ts = np.arange(nsamp)
        plt.plot(ts, wfs[iwf], '-r', label='raw')
        # plt.plot(ts, wfs_minus_ramp[iwf], '-b', label='wf-ramp')
        # plt.plot(ts, wfs_minus_ft_and_ramp[iwf], '-g', label='wf-ft-ramp')
        # plt.plot(ts, wfs_minus_trap[iwf], '-m', label='wf-ft-2ramp')
        plt.plot(ts, scratch[iwf], '-k', label='scratch')
        plt.plot(ts, trap_wfs[iwf], '-g', label='trap')

        # trapwf = pt.trap_filter(wfs[iwf], calcs)
        # plt.plot(np.arange(len(trapwf)), trapwf, '-k', label='bentrap')

        plt.legend()
        plt.show()
        exit()

    return {"wf_trap": trap_wfs}


def current_trap(waves, calcs, sigma=3, test=False):
    """
    calculate the current trace,
    by convolving w/ first derivative of a gaussian.
    """
    wfs = waves["wf_blsub"]

    wfc = gaussian_filter1d(wfs, sigma=sigma, order=1) # lol, so simple

    if test:
        iwf = 5
        ts = np.arange(len(wfs[iwf]))
        wf = wfs[iwf] / np.amax(wfs[iwf])
        curr = wfc[iwf] / np.amax(wfc[iwf])

        plt.plot(ts, wf, c='b')
        plt.plot(ts, curr, c='r', alpha=0.7)
        plt.show()

    return {"wf_current" : wfc}


def pz_correct(waves, calcs, rc, f_samp, test=False):
    """
    pole-zero correct a waveform
    """
    wfs = waves["wf_blsub"]

    if test:
        print("hi clint")
        exit()

    # # get the linear filter parameters.  RC params are in us
    # num, den = rc_decay(rc, f_samp)
    #
    # # reversing num and den does the inverse transform (ie, PZ corrects)
    # return signal.lfilter(den, num, waveform)


