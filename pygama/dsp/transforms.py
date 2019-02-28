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
        plt.xlabel("clock ticks", ha='right', x=1)
        plt.ylabel("ADC", ha='right', y=1)
        plt.legend()
        plt.tight_layout()
        plt.show()
        exit()

    return {"wf_blsub": blsub_wfs} # note, floats are gonna take up more memory


def trap_filter(waves, calcs, rise, flat, clk, decay=0, test=False):
    """
    vectorized trapezoid filter.
    input units are Hz (clk) and nanoseconds (rise, flat, decay)
    """
    # start w/ baseline-subtracted wfs
    wfs = waves["wf_blsub"]
    nwfs, nsamp = wfs.shape[0], wfs.shape[1]

    # convert params to units of [num samples]
    tsamp = 1e9 / clk # nanosec
    rt, ft, dt = int(rise/tsamp), int(flat/tsamp), decay/tsamp

    # calculate each component of the trap for the whole wf at once, then add 'em up
    wfs_minus_ramp = np.zeros_like(wfs)
    wfs_minus_ramp[:, rt:] = wfs[:, :nsamp - rt]

    wfs_minus_ft_and_ramp = np.zeros_like(wfs)
    wfs_minus_ft_and_ramp[:, (ft + rt):] = wfs[:, :nsamp - ft - rt]

    wfs_minus_trap = np.zeros_like(wfs)
    wfs_minus_trap[:, (ft + 2*rt):] = wfs[:, :nsamp - ft - 2*rt]

    trap_wfs = np.zeros_like(wfs)
    scratch = wfs - wfs_minus_ramp - wfs_minus_ft_and_ramp + wfs_minus_trap
    trap_wfs = np.cumsum(scratch, axis=1) / rt

    # pole-zero correct the trapezoid
    if dt != 0:
        dconst = 1. / (np.exp(1. / dt) - 1)
        tmp = np.cumsum(scratch, axis=1)
        pz_wfs = np.cumsum(tmp + dconst * scratch, axis=1) / (rt * dconst)

    if test:
        iwf = 5
        ts = np.arange(nsamp)
        plt.plot(ts, wfs[iwf], '-r', label='raw')
        # plt.plot(ts, wfs_minus_ramp[iwf], '-b', label='wf-ramp')
        # plt.plot(ts, wfs_minus_ft_and_ramp[iwf], '-g', label='wf-ft-ramp')
        # plt.plot(ts, wfs_minus_trap[iwf], '-m', label='wf-ft-2ramp')
        plt.plot(ts, scratch[iwf], '-k', label='scratch')
        plt.plot(ts, trap_wfs[iwf], '-g', label='trap')
        plt.plot(ts, pz_wfs[iwf], '-b', label='pz_trap, {}'.format(dt))

        # check against ben's original function
        # import pygama.sandbox.base_transforms as pt
        # trapwf = pt.trap_filter(wfs[iwf], 400, 250, 7200)
        # plt.plot(np.arange(len(trapwf)), trapwf, '-m', label='bentrap')

        plt.xlabel("clock ticks", ha='right', x=1)
        plt.ylabel("ADC", ha='right', y=1)
        plt.legend()
        plt.tight_layout()
        plt.show()
        exit()

    if dt != 0:
        trap_wfs = pz_wfs

    return {"wf_trap": trap_wfs}


def current(waves, calcs, sigma=3, test=False):
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

        plt.plot(ts, wf, c='r', alpha=0.7, label='raw wf')
        plt.plot(ts, curr, c='b', label='current')

        # compare w/ MGDO current
        from ROOT import std, MGTWaveform, MGWFTrapSlopeFilter
        tsf = MGWFTrapSlopeFilter()
        tsf.SetPeakingTime(1)
        tsf.SetIntegrationTime(10)
        tsf.SetEvaluateMode(7)
        mgwf_in, mgwf_out = MGTWaveform(), MGTWaveform()
        tmp = std.vector("double")(len(wf))
        for i in range(len(wf)): tmp[i] = wf[i]
        mgwf_in.SetData(tmp)
        tmp = mgwf_in.GetVectorData()
        tsf.TransformOutOfPlace(mgwf_in, mgwf_out)
        out = mgwf_out.GetVectorData()
        mgawf = np.fromiter(out, dtype=np.double, count=out.size())
        mgawf = mgawf / np.amax(mgawf)
        plt.plot(ts, mgawf, '-g', alpha=0.7, label='mgdo')

        plt.xlabel("clock ticks", ha='right', x=1)
        plt.ylabel('ADC', ha='right', y=1)
        plt.legend()
        plt.tight_layout()
        plt.show()
        exit()

    return {"wf_current" : wfc}


def pz_correct(waves, calcs, decay, clk, test=False):
    """
    pole-zero correct a waveform
    decay is in ns, clk is in Hz
    """
    wfs = waves["wf_blsub"]

    # get linear filter parameters
    tsamp = 1e9 / clk # ns
    dt = decay / tsamp # [number of clock ticks]
    rc = 1. / np.exp(1. / dt)
    num, den = [1, -1], [1, -rc]

    # reversing num and den does the inverse transform (ie, PZ corrects)
    pz_wfs = lfilter(den, num, wfs)

    if test:
        iwf = 5
        ts = np.arange(len(wfs[iwf]))
        plt.plot(ts, wfs[iwf], '-r', label='raw')
        plt.plot(ts, pz_wfs[iwf], '-b', label='pz_corr')

        # let's try calling the trapezoid w/ no decay time & make sure
        # the two transforms are equivalent!

        # call the trapezoid w/ no PZ correction, on THIS PZ-corrected wf
        tmp = trap_filter({"wf_blsub":pz_wfs}, calcs, 4000, 2500, 100e6)
        trap = tmp["wf_trap"][iwf]
        plt.plot(ts, trap, '-g', lw=3, label='trap on pzcorr wf')

        # compare to the PZ corrected trap on the RAW wf.
        tmp = trap_filter(waves, calcs, 4000, 2500, 100e6, 72000)
        trap2 = tmp["wf_trap"][iwf]
        plt.plot(ts, trap2, '-m', label="pz trap on raw wf")

        plt.xlabel("clock ticks", ha='right', x=1)
        plt.ylabel("ADC", ha='right', y=1)
        plt.legend()
        plt.tight_layout()
        plt.show()
        exit()

    return {"wf_pz" : pz_wfs}

