import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
import scipy.signal as signal

def blsub(waves, calcs, wfin="waveform", wfout="wf_blsub", test=False):
    """
    return an ndarray of baseline-subtracted waveforms,
    using the results from the fit_bl calculator
    """
    wfs = waves[wfin]
    nwfs, nsamp = wfs.shape[0], wfs.shape[1]

    bl_0 = calcs["bl_int"].values[:, np.newaxis]

    slope_vals = calcs["bl_slope"].values[:, np.newaxis]
    bl_1 = np.tile(np.arange(nsamp), (nwfs, 1)) * slope_vals

    blsub_wfs = wfs - (bl_0 + bl_1)

    if test:
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


def trap(waves, calcs, rise, flat, clk, decay=0, fall=None,
         wfin="wf_blsub", wfout="wf_trap", test=False):
    """
    vectorized trapezoid filter.
    inputs are in Hz (clk) and microseconds (rise, flat, decay)
    """
    # start w/ baseline-subtracted wfs
    wfs = waves[wfin]
    nwfs, nsamp = wfs.shape[0], wfs.shape[1]

    # convert params to units of [num samples]
    tsamp = 1e9 / clk  # nanosec
    rt, ft = int(rise * 1000 / tsamp), int(flat * 1000 / tsamp)
    flt = int(fall * 1000 / tsamp) if fall != None else rt
    dt = decay * 1000 / tsamp

    # calculate each component of the trap for the whole wf at once, then add 'em up
    wfs_minus_ramp = np.zeros_like(wfs)
    wfs_minus_ramp[:, rt:] = wfs[:, :nsamp - rt]

    wfs_minus_ft_and_ramp = np.zeros_like(wfs)
    wfs_minus_ft_and_ramp[:, (ft + rt):] = wfs[:, :nsamp - ft - rt]

    wfs_minus_trap = np.zeros_like(wfs)
    wfs_minus_trap[:, (ft + 2 * rt):] = wfs[:, :nsamp - ft - 2 * rt]
    # wfs_minus_trap[:, (rt + ft + flt):] = wfs[:, :nsamp - rt - ft - flt]

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

        iwf = -1
        while True:
            if iwf != -1:
                inp = input()
                if inp == "q": exit()
                if inp == "p": iwf -= 2
                if inp.isdigit() : iwf = int(inp)-1
            iwf += 1
            print(iwf)

            ts = np.arange(nsamp)

            plt.cla()
            plt.plot(ts, wfs[iwf], '-k', alpha=0.7, label='raw')
            # plt.plot(ts, wfs_minus_ramp[iwf], '-b', label='wf-rt')
            # plt.plot(ts, wfs_minus_ft_and_ramp[iwf], '-g', label='wf-rt-ft')
            # plt.plot(ts, wfs_minus_trap[iwf], '-m', label='wf-rt-ft-flt')
            # plt.plot(ts, scratch[iwf], '-k', label='scratch')
            plt.plot(ts, trap_wfs[iwf], '-g', label='trap')
            plt.plot(ts, pz_wfs[iwf], '-b', label='pz_trap, {}'.format(dt))

            # check against ben's original function
            # import pygama.sandbox.base_transforms as pt
            # trapwf = pt.trap_filter(wfs[iwf], 400, 250, 7200)
            # plt.plot(np.arange(len(trapwf)), trapwf, '-m', label='bentrap')

            # check the asym trap.  should help find t0
            # GAT WFA settings:  0.04 us ramp, 0.1 us flat, 2 us fall
            arise = int(0.04 * 1000 / tsamp)
            aflat = int(0.1 * 1000 / tsamp)
            afall = int(2 * 1000 / tsamp)
            wf = wfs[iwf]
            atrap = np.zeros(len(wf))
            w1 = int(arise)
            w2 = int(arise + aflat)
            w3 = int(arise + aflat + afall)
            for i in range(len(wf) - 1000):
                r1 = np.sum(wf[i:w1 + i]) / arise
                r2 = np.sum(wf[w2 + i:w3 + i]) / afall
                atrap[i] = r2 - r1

            # plt.plot(ts, atrap, '-m', label="asym trap")

            plt.xlabel("clock ticks", ha='right', x=1)
            plt.ylabel("ADC", ha='right', y=1)
            plt.legend()
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.001)

    if dt != 0:
        trap_wfs = pz_wfs

    return {wfout: trap_wfs}


def pz(waves, calcs, decay, clk, wfin="wf_blsub", wfout="wf_pz", test=False):
    """
    pole-zero correct a waveform
    decay is in us, clk is in Hz
    """
    wfs = waves[wfin]

    # get linear filter parameters
    tsamp = 1e9 / clk  # ns
    dt = decay*1000 / tsamp  # [number of clock ticks]
    rc = 1. / np.exp(1. / dt)
    num, den = [1, -1], [1, -rc]

    # reversing num and den does the inverse transform (ie, PZ corrects)
    pz_wfs = signal.lfilter(den, num, wfs)

    if test:
        iwf = 5
        ts = np.arange(len(wfs[iwf]))
        plt.plot(ts, wfs[iwf], '-r', label='raw')
        plt.plot(ts, pz_wfs[iwf], '-b', label='pz_corr')

        # let's try calling the trapezoid w/ no decay time & make sure
        # the two transforms are equivalent!

        # call the trapezoid w/ no PZ correction, on THIS PZ-corrected wf
        tmp = trap({"wf_blsub": pz_wfs}, calcs, 4, 2.5, 100e6)
        wf_trap = tmp["wf_trap"][iwf]
        plt.plot(ts, wf_trap, '-g', lw=3, label='trap on pzcorr wf')

        # compare to the PZ corrected trap on the RAW wf.
        tmp = trap(waves, calcs, 4, 2.500, 100e6, 72)
        wf_trap2 = tmp["wf_trap"][iwf]
        plt.plot(ts, wf_trap2, '-m', label="pz trap on raw wf")

        plt.xlabel("clock ticks", ha='right', x=1)
        plt.ylabel("ADC", ha='right', y=1)
        plt.legend()
        plt.tight_layout()
        plt.show()
        exit()

    return {wfout: pz_wfs}


def current(waves, calcs, sigma=3, wfin="wf_blsub", wfout="wf_curr", test=False):
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

        # compare w/ MGDO current, using GAT WFA parameters
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


def peakdet(waves, calcs, delta, i_end, sigma=0, wfin="wf_curr", wfout="wf_maxc", test=False):
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
    wfstd = np.std(wfc[:, :i_end], axis=1)

    # column arrays
    find_max = np.ones(wfc.shape[0])  # 1: True, 0: False
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
        if sigma > 0:
            idx2 = np.where(maxes[idxT] > (sigma * wfstd[idxT]))
            rows = idxT[0][idx2]
            cols = imaxes[idxT][idx2]

            # handle maxes
            wfmax[rows, cols] = maxes[idxT][idx2]
        else:
            wfmax[idxT[0], imaxes[idxT]] = maxes[idxT]

        mins[idxT] = maxes[idxT]
        find_max[idxT] = False

        # handle mins
        wfmin[idxF[0], imins[idxF]] = mins[idxF]
        maxes[idxF] = mins[idxF]
        find_max[idxF] = True

    if test:

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
            from pygama.utils import peakdet as eli_peakdet
            maxpks, minpks = eli_peakdet(wfc[iwf], delta)

            # crude zoom into the rising edge
            # t50 = np.argmax(awf)
            # tlo, thi = t50 - 300, t50 + 300
            # ts, wf, awf = ts[tlo:thi], wf[tlo:thi], awf[tlo:thi]

            plt.cla()
            plt.plot(ts, wf, '-b', alpha=0.7, label='data')
            plt.plot(ts, awf, '-k', label='current')

            if sigma != 0:
                plt.axhline(
                    sigma * wfstd[iwf] / np.amax(wfc[iwf]),
                    c='g',
                    lw=2,
                    label="{} sigma".format(sigma))
            plt.axvline(i_end, c='r', alpha=0.7, label="bl avg window")

            # peakdet peaks
            for i, idx in enumerate(np.where(wfmax[iwf] > 0)[0]):
                label = "peakdet" if i == 0 else None
                plt.plot(
                    idx,
                    wfmax[iwf][idx] / np.amax(wfc[iwf]),
                    ".m",
                    ms=20,
                    label=label)

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


def peakdet_test(waves, calcs, delta, sigma, i_end, test=False):
    """
    do a speed test of the two peakdet methods
    """
    import time
    from pygama.utils import peakdet as eli_peakdet

    start = time.time()
    print("sigma is", sigma)
    tmp = peakdet(waves, calcs, delta, i_end, sigma)
    tmp1 = tmp["wf_maxc"]
    print(
        "vectorized took {:.4f} sec.  tmp1 shape:".format(time.time() - start),
        tmp1.shape)

    start = time.time()
    wfc = waves["wf_curr"]
    tmp2 = np.zeros_like(wfc)
    for i, wf in enumerate(wfc):

        maxpks, minpks = eli_peakdet(wf, delta)

        # could try np.vectorize here
        # vfunc = np.vectorize(eli_peakdet)
        # maxpks, minpks = vfunc(wf, delta) # this fails, i'm probably doing it wrong

        if len(maxpks) > 0:
            rows = np.full(len(maxpks), i)
            cols = maxpks[:, 0].astype(int)
            tmp2[rows, cols] = maxpks[:, 1]

    print("regular took {:.4f} sec.  tmp2 shape:".format(time.time() - start),
          tmp2.shape)

    inomatch = np.where(tmp1 != tmp2)
    print("num entries total: {}  not matching {}".format(
        tmp1.size, len(inomatch[0])))

    # print("vect values:")
    # print(tmp1[inomatch])
    # print("reg values:")
    # print(tmp2[inomatch])

    print("are they equal?", np.all(tmp1 == tmp2))

    # so i find a few minor differences, but not really enough to be alarming.
    # they're probably related to not matching the find_max condition
    # quite correctly to the original peakdet function.
    # for a wf block of (200,2999), i get ~0.6 seconds for old method,
    # ~0.1 for the new one, so a factor 6 faster. probably even better on
    # larger wf blocks.
    # ok, i hereby bless the "partially vectorized" peakdet function.

    exit()


def savgol(waves, calcs, window=47, order=2, wfin="wf_blsub", wfout="wf_savgol", test=False):
    """
    apply a savitzky-golay filter to a wf.
    this is good for reducing noise on e.g. timepoint calculations
    """
    wfs = waves[wfin]

    # Silence harmless warning you get using savgol on old LAPACK
    # import warnings
    # warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

    wfsg = signal.savgol_filter(wfs, window, order)

    if test:
        iwf = 4
        ts = np.arange(len(wfs[iwf]))
        plt.plot(ts, wfs[iwf], '-b', label='raw')
        plt.plot(ts, wfsg[iwf], '-r', label='savgol')
        plt.xlabel("clock ticks", ha='right', x=1)
        plt.ylabel("adc", ha='right', y=1)
        plt.legend()
        plt.tight_layout()
        plt.show()
        exit()

    return {wfout: wfsg}


def psd(waves, calcs, nseg=100, test=False):
    """
    calculate the psd of a bunch of wfs, and output them as a block,
    so some analysis can add them all together.
    nperseg = 1000 has more detail, but is slower
    """
    wfs = waves["wf_blsub"]

    f, p = signal.welch(wfs, 100e6, nperseg=nseg)

    if test:

        plt.semilogy(f, p[3], '-k', alpha=0.4, label='one wf')

        ptot = np.sum(p, axis=0)
        plt.semilogy(f, ptot / wfs.shape[0], '-b', label='all wfs')

        plt.xlabel('Frequency (Hz)', ha='right', x=0.9)
        plt.ylabel('PSD (ADC^2 / Hz)', ha='right', y=1)
        plt.legend(loc=1)
        plt.tight_layout()
        plt.show()
        exit()

    return {"psd": p, "f_psd": f}


def notch(waves, calcs, f_notch, Q, clk, wfin="wf_blsub", wfout="wf_notch", test=False):
    """
    apply notch filter with some quality factor Q
    """
    wfs = waves[wfin]

    f_nyquist = 0.5 * clk
    num, den = signal.iirnotch(f_notch / f_nyquist, Q)
    wf_notch = signal.lfilter(num, den, wfs)

    if test:

        iwf = -1
        while True:
            if iwf != -1:
                inp = input()
                if inp == "q": exit()
                if inp == "p": iwf -= 2
                if inp.isdigit() : iwf = int(inp)-1
            iwf += 1
            print(iwf)

            ts = np.arange(len(wfs[iwf]))

            plt.cla()
            plt.plot(ts, wfs[iwf], "-b", label='raw wf')
            plt.plot(ts, wf_notch[iwf], "-r",
                     label="notch wf, f {:.1e}, Q {}".format(f_notch, Q))

            plt.xlabel("clock ticks", ha='right', x=1)
            plt.ylabel("ADC", ha='right', y=1)
            plt.legend()
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.01)

    return {wfout : wf_notch}


def center(waves, calcs, tp=50, n_pre=150, n_post=150,
           wfin="wf_savgol", wfout="wf_ctr", test=False):
    """
    return a waveform centered (windowed) around i_ctr.
    default: center around the 50 pct timepoint
    TODO:
    - probably need to have some error handling if the tp fails, or the
      indexes for the window go out of bounds
    - should we try to preseve the original timestamps?  nah, probably more
      trouble than it's worth.
    """
    wfs = waves[wfin]
    tps = calcs["tp{}".format(tp)].values

    # select the indexes we want to keep
    # ugh, should figure out how to do this w/o the loop
    wf_idxs = np.zeros([wfs.shape[0], n_pre + n_post], dtype=int)
    row_idxs = np.zeros_like(wf_idxs)
    for i, tp in enumerate(tps):
        wf_idxs[i, :] = np.arange(tp - n_pre, tp + n_post)
        row_idxs[i, :] = i

    # apply the selection
    wf_ctr = wfs[row_idxs, wf_idxs]

    if test:
        ts = np.arange(wf_ctr.shape[1])
        for wf in wf_ctr:
            if 200 < np.amax(wf) < 2000:
                plt.plot(ts, wf, "-", lw=1)

        plt.axvline(n_pre+1, c='k', label="")
        plt.xlabel("clock ticks (shifted)", ha='right', x=1)
        plt.ylabel("ADC", ha='right', y=1)
        plt.tight_layout()
        plt.show()
        exit()

    return {wfout : wf_ctr}


def trim(waves, calcs, n_pre, n_post, wfin="wf_blsub", wfout="wf_trim", test=False):
    """
    cut out the first n_pre and the last n_post samples.
    """
    wfs = waves[wfin]

    wf_trim = wfs[:,n_pre:n_post]

    if test:
        iwf = 5
        ts = np.arange(wf_trim.shape[1])
        plt.plot(ts, wf_trim[iwf])
        plt.xlabel("clock ticks", ha='right', x=1)
        plt.ylabel("ADC", ha='right', y=1)
        plt.tight_layout()
        plt.show()
        exit()

    return {wfout : wf_trim}


def interp(waveform, offset):
    """
    allows you to shift a waveform's timestamps by an amount
    less than the clock frequency (i.e. allow them to float in a fit,
    without discretizing them.)
    TODO:
    - vectorize this, but only do that when we actually need this function.
      maybe we could use it with the timepoint calculator? if we can't
      handle 10ns uncertainty on a timepoint.
    """
    xp = np.arange(len(waveform))
    x = xp[:-1] + offset
    return np.interp(x, xp, waveform)


def asym_trap(waveform, ramp=200, flat=100, fall=40, padAfter=False):
    """
    compute an asymmetric trapezoidal filter
    TODO: can you just add fall=None to the trap function and combine this?
    """
    trap = np.zeros(len(waveform))
    for i in range(len(waveform) - 1000):
        w1 = ramp
        w2 = ramp + flat
        w3 = ramp + flat + fall
        r1 = np.sum(waveform[i:w1 + i]) / (ramp)
        r2 = np.sum(waveform[w2 + i:w3 + i]) / (fall)
        if not padAfter:
            trap[i + 1000] = r2 - r1
        else:
            trap[i] = r2 - r1
    return trap


def nlc(waveform, time_constant_samples, fNLCMap, fNLCMap2=None, n_bl=100):
    """
    """
    map_offset = np.int((len(fNLCMap) - 1) / 2)

    if (time_constant_samples == 0.):
        waveform -= fNLCMap.GetCorrection()
    else:
        # Apply Radford's time-lagged correction

        # first, average baseline at the beginning to get a good starting
        # point for current_inl

        if (n_bl >= len(waveform)):
            print("input wf length is only {}".format(len(waveform)))
            return

        # David initializes bl to nBLEst/2 to get good rounding when he does integer division
        # by nBLEst, but we do floating point division so that's not necessary

        try:
            bl = np.int(np.sum(waveform[:n_bl]) / n_bl)
            current_inl = fNLCMap[bl + map_offset]

            for i in range(n_bl, len(waveform)):
                wf_pt_int = np.int(waveform[i])
                if wf_pt_int + map_offset >= len(fNLCMap): continue

                summand = fNLCMap[wf_pt_int + map_offset]
                current_inl += (summand - current_inl) / time_constant_samples
                if (fNLCMap2 is None): waveform[i] -= current_inl
                else:
                    waveform[i] -= fNLCMap2[wf_pt_int +
                                            map_offset] + current_inl
                    # maybe needs to be +=! check

        except IndexError:
            print("\nadc value {} and int {}".format(waveform[i], wf_pt_int))
            print("wf_offset {}".format(map_offset))
            print("looking for index {}/{}".format(wf_pt_int + map_offset,
                                                   len(fNLCMap)))

    return waveform
