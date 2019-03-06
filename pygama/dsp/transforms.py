import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import lfilter


def bl_sub(waves, calcs, wfin="waveform", wfout="wf_blsub", test=False):
    """
    Return an ndarray of baseline-subtracted waveforms
    Depends on fit_baseline calculator.
    for reference, the non-vector version is just:
    return waveform - (bl_0 + bl_1 * np.arange(len(waveform)))
    """
    wfs = waves[wfin]
    nwfs, nsamp = wfs.shape[0], wfs.shape[1]

    bl_0 = calcs["bl_int"].values[:, np.newaxis]

    slope_vals = calcs["bl_slope"].values[:, np.newaxis]
    bl_1 = np.tile(np.arange(nsamp), (nwfs, 1)) * slope_vals

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

    # note, floats are gonna take up more memory
    return {wfout: blsub_wfs}


def trap_filter(waves,
                calcs,
                rise,
                flat,
                clk,
                decay=0,
                wfin="wf_blsub",
                wfout="wf_trap",
                test=False):
    """
    vectorized trapezoid filter.
    input units are Hz (clk) and nanoseconds (rise, flat, decay)
    """
    # start w/ baseline-subtracted wfs
    wfs = waves[wfin]
    nwfs, nsamp = wfs.shape[0], wfs.shape[1]

    # convert params to units of [num samples]
    tsamp = 1e9 / clk  # nanosec
    rt, ft, dt = int(rise / tsamp), int(flat / tsamp), decay / tsamp

    # calculate each component of the trap for the whole wf at once, then add 'em up
    wfs_minus_ramp = np.zeros_like(wfs)
    wfs_minus_ramp[:, rt:] = wfs[:, :nsamp - rt]

    wfs_minus_ft_and_ramp = np.zeros_like(wfs)
    wfs_minus_ft_and_ramp[:, (ft + rt):] = wfs[:, :nsamp - ft - rt]

    wfs_minus_trap = np.zeros_like(wfs)
    wfs_minus_trap[:, (ft + 2 * rt):] = wfs[:, :nsamp - ft - 2 * rt]

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

    return {wfout: trap_wfs}


def pz_correct(waves,
               calcs,
               decay,
               clk,
               wfin="wf_blsub",
               wfout="wf_pz",
               test=False):
    """
    pole-zero correct a waveform
    decay is in ns, clk is in Hz
    """
    wfs = waves[wfin]

    # get linear filter parameters
    tsamp = 1e9 / clk  # ns
    dt = decay / tsamp  # [number of clock ticks]
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
        tmp = trap_filter({"wf_blsub": pz_wfs}, calcs, 4000, 2500, 100e6)
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

    return {wfout: pz_wfs}


def current(waves,
            calcs,
            sigma=3,
            wfin="wf_blsub",
            wfout="wf_current",
            test=False):
    """
    calculate the current trace,
    by convolving w/ first derivative of a gaussian.
    """
    wfs = waves[wfin]

    wfc = gaussian_filter1d(wfs, sigma=sigma, order=1)  # lol, so simple

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
        for i in range(len(wf)):
            tmp[i] = wf[i]
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

    return {wfout: wfc}


def peakdet(waves, calcs, delta, sigma, i_end,
            wfin="wf_current", wfout="wf_maxc", test=False):
    """
    find multiple maxima in the current wfs.
    this can be optimized for multi-site events, or pile-up events.

    since this can find multiple values, we can't directly save to the calcs
    dataframe (not single-valued).  For now, let's save to a sparse wf block.
    alternately, we could encode the num/vals of the peaks we find into an
    integer, to make this a calculator.  IDK if it would be a pain to decode
    the output number.

    eli's algorithm (see pygama.utils.peakdet) is dependent on the previous
    value of the given wf.  this limits what we can vectorize.
    I think this is called a "forward-dependent" loop, but I'm not sure.
    So this version only vectorizes the operation on each column.

    TODO: implement a threshold, where we ignore peaks within 2 sigma of
    the baseline noise of the current pulse
    """
    # input and output blocks
    wfc = waves[wfin]
    wfmax = np.zeros_like(wfc)
    wfmin = np.zeros_like(wfc)

    # calculate the noise on each wf
    wfstd = np.std(wfc[:,:i_end], axis=1)

    # column arrays
    find_max = np.ones(wfc.shape[0]) # 1: True, 0: False
    maxes = np.full(wfc.shape[0], -np.inf)
    mins = np.full(wfc.shape[0], np.inf)
    imaxes = np.zeros(len(maxes), dtype=np.int)
    imins = np.zeros(len(mins), dtype=np.int)

    # scan over columns of the input block.  this loop is unavoidable
    for i, wfcol in enumerate(wfc.T):

        imax = np.where(np.greater(wfcol, maxes) == True)
        imin = np.where(np.less(wfcol, mins) == True)
        maxes[imax] = wfcol[imax]
        mins[imin] = wfcol[imin]
        imaxes[imax], imins[imin] = int(i), int(i)
        # print(i, maxes[:3]) # see vals increasing.  OK
        # print(i, mins[:3]) # see vals decreasing.  OK

        # do the where's separately, before modifying find_max
        idxT = np.where((find_max == True) & (wfcol < maxes - delta))
        idxF = np.where((find_max == False) & (wfcol > mins + delta))

        # if len(idxT[0]) > 0:
        #     print("found some maxes.")
        #     print("  idxT", idxT[0], idxT[0].dtype)
        #     print("  maxes", maxes[idxT], maxes[idxT].dtype)
        #     print("  imaxes", imaxes[idxT], imaxes[idxT].dtype)

        # only take evts above the noise (not in orig peakdet function)
        idx2 = np.where(maxes[idxT] > (sigma * wfstd[idxT]))
        rows = idxT[0][idx2]
        cols = imaxes[idxT][idx2]

        # handle maxes
        wfmax[rows, cols] = maxes[idxT][idx2]
        mins[idxT] = maxes[idxT]
        find_max[idxT] = False

        # handle mins
        wfmin[idxF[0], imins[idxF]] = mins[idxF]
        maxes[idxF] = mins[idxF]
        find_max[idxF] = True

    if test:

        from pygama.utils import peakdet as eli_peakdet
        wfs = waves["wf_blsub"]

        iwf = -1
        while True:
            if iwf != -1:
                inp = input()
                if inp == "q": exit()
                if inp == "p": iwf -= 2
            iwf += 1
            print(iwf)

            ts = np.arange(len(wfs[iwf]))
            wf = wfs[iwf] / np.amax(wfs[iwf])
            awf = wfc[iwf] / np.amax(wfc[iwf])

            # try out eli's pileup detector
            maxpks, minpks = eli_peakdet(wfc[iwf], delta)
            # print(maxpks[:,0]) # indices
            # print(maxpks[:,1]) # found values

            # crude zoom into the rising edge
            # t50 = np.argmax(awf)
            # tlo, thi = t50 - 300, t50 + 300
            # ts, wf, awf = ts[tlo:thi], wf[tlo:thi], awf[tlo:thi]

            plt.cla()
            plt.plot(ts, wf, '-b', alpha=0.7, label='data')
            plt.plot(ts, awf, '-k', label='current')

            plt.axhline(sigma * wfstd[iwf] / np.amax(wfc[iwf]), c='g', lw=2,
                        label="{} sigma".format(sigma))
            plt.axvline(i_end, c='r', alpha=0.7, label="bl avg window")

            # peakdet peaks
            for i, idx in enumerate(np.where(wfmax[iwf] > 0)[0]):
                label = "peakdet" if i==0 else None
                plt.plot(idx, wfmax[iwf][idx] / np.amax(wfc[iwf]), ".m", ms=20, label=label)

            # # eli peaks
            # for i, (idx, val) in enumerate(maxpks):
            #     label = "orig. peakdet" if i==0 else None
            #     plt.plot(idx, val / np.amax(wfc[iwf]), ".g", ms=10, label=label)

            plt.xlabel("clock ticks", ha='right', x=1)
            plt.ylabel('ADC', ha='right', y=1)
            plt.legend()
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.01)

    return {wfout: wfmax}


def peakdet_test(waves, calcs, delta, test=False):
    """
    do a speed test of the two peakdet methods
    """
    import time
    from pygama.utils import peakdet as eli_peakdet

    start = time.time()
    tmp = peakdet(waves, calcs, delta)
    tmp1 = tmp["wf_maxc"]
    print("vectorized took {:.4f} sec.  tmp1 shape:".format(time.time()-start), tmp1.shape)

    start = time.time()
    wfc = waves["wf_current"]
    tmp2 = np.zeros_like(wfc)
    for i, wf in enumerate(wfc):

        # could also try np.vectorize here
        maxpks, minpks = eli_peakdet(wf, delta)
        if len(maxpks) > 0:
            rows = np.full(len(maxpks), i)
            cols = maxpks[:,0].astype(int)
            tmp2[rows, cols] = maxpks[:,1]

    print("regular took {:.4f} sec.  tmp2 shape:".format(time.time()-start), tmp2.shape)

    inomatch = np.where(tmp1 != tmp2)
    print("vect values:")
    print(tmp1[inomatch])
    print("reg values:")
    print(tmp2[inomatch])

    print("are they equal?", np.all(tmp1 == tmp2))

    # so i find a few minor differences, but none too alarming.
    # ok, i hereby bless the "partially vectorized" peakdet function.

    exit()


def timepoint():
    """
    note: may have to do a loop over columns for the timepoint calculator too.
    """
    print('hi')


def diff():
    """
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.diff.html
    just try it.  also try higher orders.
    """
    print("hi")


def grad():
    """
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.gradient.html
    """
    print("hi")
