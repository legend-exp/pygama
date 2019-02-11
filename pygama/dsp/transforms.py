import numpy as np
import pandas as pd


def bl_subtract(waves, calcs, test=False):
    """ Return an ndarray of baseline-subtracted waveforms
    Depends on fit_baseline calcsulator.
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
        bl_avg = calcs["bl_avg"].values[:,np.newaxis]
        blsub_avgs = wfs - bl_avg

        # quick diagnostic plot
        import matplotlib.pyplot as plt
        iwf = 1
        plt.plot(np.arange(nsamp), wfs[iwf], '-r', label="raw")
        plt.plot(np.arange(nsamp), blsub_wfs[iwf], '-b', label="bl_sub")
        plt.plot(np.arange(nsamp), blsub_avgs[iwf], '-g', label="bl_avg")
        plt.legend()
        plt.show()
        exit()

    return {"wf_blsub": blsub_wfs} # note, floats are gonna take up more memory


def trap_filter(waves, calcs, rt=400, ft=200, dt=0, test=False):

    wfs = waves["wf_blsub"]
    nwfs, nsamp = wfs.shape[0], wfs.shape[1]

    wfs_minus_ramp = np.zeros_like(wfs)
    wfs_minus_ramp[:, :rt] = 0
    wfs_minus_ramp[:, rt:] = wfs[:, :nsamp - rt]

    wfs_minus_ft_and_ramp = np.zeros_like(wfs)
    wfs_minus_ft_and_ramp[:, :(ft + rt)] = 0
    wfs_minus_ft_and_ramp[:, (ft + rt):] = wfs[:, :nsamp - ft - rt]

    wfs_minus_ft_and_2ramp = np.zeros_like(wfs)
    wfs_minus_ft_and_2ramp[:, :(ft + 2 * rt)] = 0
    wfs_minus_ft_and_2ramp[:, (ft + 2 * rt):] = wfs[:, :nsamp - ft - 2 * rt]

    scratch = wfs - (wfs_minus_ramp + wfs_minus_ft_and_ramp + wfs_minus_ft_and_2ramp)

    trap_wfs = np.zeros_like(wfs)
    trap_wfs = np.cumsum(trap_wfs + scratch, axis=1) / rt

    if test:
        # diagnostic plot
        import matplotlib.pyplot as plt
        import pygama.processing.transforms as pt
        iwf = 2
        plt.plot(np.arange(nsamp), wfs[iwf], '-r', label='raw')
        # plt.plot(np.arange(nsamp), wfs_minus_ramp[iwf], '-b', label='wf-ramp')
        # plt.plot(np.arange(nsamp), wfs_minus_ft_and_ramp[iwf], '-g', label='wf-ft-ramp')
        # plt.plot(np.arange(nsamp), wfs_minus_ft_and_2ramp[iwf], '-m', label='wf-ft-2ramp')
        # plt.plot(np.arange(nsamp), scratch[iwf], '-b', label='scratch')
        plt.plot(np.arange(nsamp), trap_wfs[iwf], '-g', lw=4, label='trap')

        trapwf = pt.trap_filter(wfs[iwf])
        plt.plot(np.arange(len(trapwf)), trapwf, '-k', label='bentrap')

        plt.ylim(-1000, 1.2*np.amax(wfs[iwf]))
        plt.legend()
        plt.show()
        exit()

    return {"wf_trap": trap_wfs}

