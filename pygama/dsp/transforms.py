import time
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.ndimage as ndimage

# silence harmless warnings
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
warnings.filterwarnings(action="ignore", module="scipy.signal", category=FutureWarning)

def blsub(waves, calcs, blest="", wfin="waveform", wfout="wf_blsub", test=False):
    """
    return an ndarray of baseline-subtracted waveforms,
    using the results from the fit_bl calculator
    """
    wfs = waves[wfin]
    nwfs, nsamp = wfs.shape[0], wfs.shape[1]

    if blest == "fcdaq":
        bl_0 = calcs["fcdaq"].values[:, np.newaxis]
        blsub_wfs = wfs - bl_0
    
    else:
        bl_0 = calcs["bl_p0"].values[:, np.newaxis]
        if "bl_p1" in calcs.keys():
            slope_vals = calcs["bl_p1"].values[:, np.newaxis]
            bl_1 = np.tile(np.arange(nsamp), (nwfs, 1)) * slope_vals
            blsub_wfs = wfs - (bl_0 + bl_1)
        else:
            blsub_wfs = wfs - bl_0

    if test:
      iwf = 2
      while True:
        if iwf != 2:
          inp = input()
          if inp == "q": exit()
          if inp == "p": iwf -= 2
          if inp.isdigit(): iwf = int(inp) - 1
        iwf += 1
        print(iwf)
        plt.cla()

        plt.plot(np.arange(nsamp), wfs[iwf], '-g', label="raw")
        plt.plot(np.arange(nsamp), blsub_wfs[iwf], '-b', label="bl_sub")
        # plt.plot(np.arange(nsamp), blsub_avgs[iwf], '-g', label="bl_avg")

        plt.xlabel("clock ticks", ha='right', x=1)
        plt.ylabel("ADC", ha='right', y=1)
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)
        plt.grid(True)
        plt.pause(0.01)

    # note, floats are gonna take up more memory
    return {wfout: blsub_wfs}


def trap(waves, calcs, rise, flat, fall=None, decay=0, wfin="wf_blsub", wfout="wf_trap", test=False):
    """
    vectorized trapezoid filter.
    inputs are in microsec (rise, flat, fall, decay)
    """
    wfs = waves[wfin]
    clk = waves["settings"]["clk"] # Hz

    # convert params to units of [num samples, i.e. clock ticks]
    nsamp = 1e10 / clk
    rt, ft, dt = int(rise * nsamp), int(flat * nsamp), decay * nsamp
    flt = rt if fall is None else int(fall * nsamp)

    # calculate trapezoids
    if rt == flt:
        """
        symmetric case, use recursive trap (fastest)
        """
        tr1, tr2, tr3 = np.zeros_like(wfs), np.zeros_like(wfs), np.zeros_like(
            wfs)
        tr1[:, rt:] = wfs[:, :-rt]
        tr2[:, (ft + rt):] = wfs[:, :-rt - ft]
        tr3[:, (rt + ft + flt):] = wfs[:, :-rt - ft - flt]
        scratch = (wfs - tr1) - (tr2 - tr3)
        atrap = np.cumsum(scratch, axis=1) / rt
    else:
        """
        asymmetric case, use the fastest non-recursive algo i could find.
        (I also tried scipy.ndimage.convolve1d, scipy.signal.[fft]convolve)
        TODO (someday): change this to be recursive (need to math it out)
        https://www.sciencedirect.com/science/article/pii/0168900294910111
        """
        kernel = np.zeros(rt + ft + flt)
        kernel[:rt] = 1 / rt
        kernel[rt + ft:] = -1 / flt
        atrap = np.zeros_like(wfs)  # faster than a list comprehension
        for i, wf in enumerate(wfs):
            atrap[i, :] = np.convolve(wf, kernel, 'same')
        npad = int((rt+ft+flt)/2)
        atrap = np.pad(atrap, ((0, 0), (npad, 0)), mode='constant')[:, :-npad]
        # atrap[:, -(npad):] = 0

    # pole-zero correct the trapezoids
    if dt != 0:
        rc = 1 / np.exp(1 / dt)
        num, den = [1, -1], [1, -rc]
        ptrap = signal.lfilter(den, num, atrap)

    if test:
        iwf = 2
        while True:
            if iwf != 2:
                inp = input()
                if inp == "q": exit()
                if inp == "p": iwf -= 2
                if inp.isdigit(): iwf = int(inp) - 1
            iwf += 1
            print(iwf)
            plt.cla()

            wf, ts = wfs[iwf], np.arange(wfs[iwf].shape[0])
            plt.plot(ts, wf, '-b', lw=2, alpha=0.7, label='raw wf')
            plt.plot(ts, atrap[iwf], '-r', label='trap')

            if dt != 0:
                plt.plot(ts, ptrap[iwf], '-k', label='pz')

            plt.xlabel("clock ticks", ha='right', x=1)
            plt.ylabel("ADC", ha='right', y=1)
            plt.legend()
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.00001)

    if dt != 0:
        return {wfout: ptrap}
    else:
        return {wfout: atrap}


def pz(waves, calcs, decay, wfin="wf_blsub", wfout="wf_pz", test=False):
    """
    pole-zero correct a waveform
    decay is in us, clk is in Hz
    """
    wfs = waves[wfin]
    clk = waves["settings"]["clk"]

    # get linear filter parameters, in units of [clock ticks]
    dt = decay * (1e10 / clk)
    rc = 1 / np.exp(1 / dt)
    num, den = [1, -1], [1, -rc]

    # reversing num and den does the inverse transform (ie, PZ corrects)
    pz_wfs = signal.lfilter(den, num, wfs)

    if test:
        iwf = -1
        while True:
            if iwf != -1:
                inp = input()
                if inp == "q": exit()
                if inp == "p": iwf -= 2
            iwf += 1
            print(iwf)

            plt.cla()
            ts = np.arange(len(wfs[iwf]))
            plt.plot(ts, wfs[iwf], '-r', label='raw')
            plt.plot(ts, pz_wfs[iwf], '-b', label='pz_corr')

            # let's try calling the trapezoid w/ no decay time & make sure
            # the two transforms are equivalent!

            # call the trapezoid w/ no PZ correction, on THIS PZ-corrected wf
            tmp = trap({"wf_blsub": pz_wfs, "settings":waves["settings"]}, calcs, 4, 2.5)
            wf_trap = tmp["wf_trap"][iwf]
            plt.plot(ts, wf_trap, '-g', lw=3, label='trap on pzcorr wf')

            # compare to the PZ corrected trap on the RAW wf.
            tmp = trap(waves, calcs, 4, 2.5, decay=72)
            wf_trap2 = tmp["wf_trap"][iwf]
            plt.plot(ts, wf_trap2, '-m', label="pz trap on raw wf")

            plt.xlabel("clock ticks", ha='right', x=1)
            plt.ylabel("ADC", ha='right', y=1)
            plt.legend()
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.01)

    return {wfout: pz_wfs}


def current(waves, calcs, sigma, wfin="wf_blsub", wfout="wf_current", test=False):
    """
    calculate the current trace,
    by convolving w/ first derivative of a gaussian.
    """
    wfs = waves[wfin]

    wfc = ndimage.filters.gaussian_filter1d(wfs, sigma=sigma, order=1) # lol

    if test:
        iwf = -1
        while True:
            if iwf != -1:
                inp = input()
                if inp == "q": exit()
                if inp == "p": iwf -= 2
            iwf += 1
            print(iwf)
            plt.cla()

            ts = np.arange(len(wfs[iwf]))
            wf = wfs[iwf] / np.amax(wfs[iwf])
            curr = wfc[iwf] / np.amax(wfc[iwf])

            # plt.plot(ts, wf, c='r', alpha=0.7, label='raw wf')
            # plt.plot(ts, curr, c='b', label='current')

            # super crude window
            idx = np.where((ts > 800) & (ts < 1200))
            plt.plot(ts[idx], wf[idx], c='r', alpha=0.7, label='raw wf')
            plt.plot(ts[idx], curr[idx], c='b', label='current')

            # # compare w/ MGDO current, using GAT WFA parameters
            # from ROOT import std, MGTWaveform, MGWFTrapSlopeFilter
            # tsf = MGWFTrapSlopeFilter()
            # tsf.SetPeakingTime(1)
            # tsf.SetIntegrationTime(10)
            # tsf.SetEvaluateMode(7)
            # mgwf_in, mgwf_out = MGTWaveform(), MGTWaveform()
            # tmp = std.vector("double")(len(wf))
            # for i in range(len(wf)):
            #     tmp[i] = wf[i]
            # mgwf_in.SetData(tmp)
            # tmp = mgwf_in.GetVectorData()
            # tsf.TransformOutOfPlace(mgwf_in, mgwf_out)
            # out = mgwf_out.GetVectorData()
            # mgawf = np.fromiter(out, dtype=np.double, count=out.size())
            # mgawf = mgawf / np.amax(mgawf)
            # plt.plot(ts, mgawf, '-g', alpha=0.7, label='mgdo')

            plt.xlabel("clock ticks", ha='right', x=1)
            plt.ylabel('ADC', ha='right', y=1)
            plt.legend()
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.01)

    return {wfout: wfc}


def peakdet(waves, calcs, delta, ihi, sigma=0, wfin="wf_current", wfout="wf_maxc", test=False):
    """
    find multiple maxima in the current wfs.
    this can be optimized for multi-site events, or pile-up events.

    eli's algorithm (see pygama.utils.peakdet) is dependent on the previous
    value of the given wf, with multiple true/false statements.
    this limits what we can vectorize.
    I think this is called a "forward-dependent" loop, but I'm not sure.
    So this version only vectorizes the operation on each column.

    this routine also uses a threshold, where we ignore peaks within 2 sigma of
    the baseline noise of the current pulse

    since this can find multiple values, we can't directly save to the calcs
    dataframe (not single-valued).  For now, let's save to a sparse wf block.
    alternately, we could encode the num/vals of the peaks we find into an
    integer, to make this a calculator.  IDK if it would be a pain to decode
    the output number.
    """
    # input and output blocks
    wfc = waves[wfin]
    wfmax = np.zeros_like(wfc)
    wfmin = np.zeros_like(wfc)

    # calculate the noise on each wf
    wfstd = np.std(wfc[:, :ihi], axis=1)

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

            ts = np.arange(len(wfc[iwf]))
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
            # plt.plot(ts, wf, '-b', alpha=0.7, label='data')
            plt.plot(ts, awf, '-k', label='current')

            if sigma != 0:
                plt.axhline(
                    sigma * wfstd[iwf] / np.amax(wfc[iwf]),
                    c='g',
                    lw=2,
                    label="{} sigma".format(sigma))
            plt.axvline(ihi, c='r', alpha=0.7, label="bl avg window")

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
            plt.pause(0.001)

    return {wfout: wfmax}


def savgol(waves, calcs, window=47, order=2, wfin="wf_blsub", wfout="wf_savgol", test=False):
    """
    apply a savitzky-golay filter to a wf.
    this is good for reducing noise on e.g. timepoint calculations
    """
    wfs = waves[wfin]

    wfsg = signal.savgol_filter(wfs, window, order)

    if test:
      iwf = -1
      while True:
        if iwf != -1:
          inp = input()
          if inp == "q": exit()
          if inp == "p": iwf -= 2
        iwf += 1
        print(iwf)
        plt.cla()

        ts = np.arange(len(wfs[iwf]))
        plt.plot(ts, wfs[iwf], '-b', label='raw')
        plt.plot(ts, wfsg[iwf], '-r', label='savgol')
        plt.xlabel("clock ticks", ha='right', x=1)
        plt.ylabel("adc", ha='right', y=1)
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)
            
    return {wfout: wfsg}


def psd(waves, calcs, ilo=None, ihi=None, nseg=100, test=False):
    """
    calculate the psd of a bunch of wfs, and output them as a block,
    so some analysis can add them all together.
    nperseg = 1000 has more detail, but is slower
    """
    wfs = waves["wf_blsub"]
    if ilo is not None and ihi is not None:
        wfs = wfs[:, ilo:ihi]
    clk = waves["settings"]["clk"] # Hz

    nseg = 2999
    f, p = signal.welch(wfs, clk, nperseg=nseg)

    if test:

        # plt.semilogy(f, p[3], '-k', alpha=0.4, label='one wf')

        ptot = np.sum(p, axis=0)
        y = ptot / wfs.shape[0]
        plt.semilogy(f, y, '-b', label='all wfs')

        plt.xlabel('Frequency (Hz)', ha='right', x=0.9)
        plt.ylabel('PSD (ADC^2 / Hz)', ha='right', y=1)
        plt.legend(loc=1)
        plt.tight_layout()
        plt.show()
        np.savez("./psd_stuff1.npz", f, y)
        exit()

    return {"psd": p, "f_psd": f}


def notch(waves, calcs, f_notch, Q, wfin="wf_blsub", wfout="wf_notch", test=False):
    """
    apply notch filter with some quality factor Q
    TODO: apply multiple notches (f_notch, Q could be lists)
    """
    wfs = waves[wfin]
    clk = waves["settings"]["clk"] # Hz

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
                if inp.isdigit(): iwf = int(inp) - 1
            iwf += 1
            print(iwf)

            ts = np.arange(len(wfs[iwf]))

            plt.cla()
            plt.plot(ts, wfs[iwf], "-b", label='raw wf')
            plt.plot(
                ts,
                wf_notch[iwf],
                "-r",
                label="notch wf, f {:.1e}, Q {}".format(f_notch, Q))

            plt.xlabel("clock ticks", ha='right', x=1)
            plt.ylabel("ADC", ha='right', y=1)
            plt.legend()
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.01)

    return {wfout : wf_notch}


def center(waves, calcs, tp=50, n_pre=150, n_post=150, wfin="wf_savgol", wfout="wf_ctr", test=False):
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
    # print(wfs.shape, row_idxs.shape, wf_idxs.shape)
    wf_ctr = wfs[row_idxs, wf_idxs]

    if test:
        ts = np.arange(wf_ctr.shape[1])
        for wf in wf_ctr:
            if 200 < np.amax(wf) < 2000:
                plt.plot(ts, wf, "-", lw=1)

        plt.axvline(n_pre + 1, c='k', label="")
        plt.xlabel("clock ticks (shifted)", ha='right', x=1)
        plt.ylabel("ADC", ha='right', y=1)
        plt.tight_layout()
        plt.show()
        exit()

    return {wfout: wf_ctr}


def trim(waves, calcs, n_pre, n_post, wfin="wf_blsub", wfout="wf_trim", test=False):
    """
    cut out the first n_pre and the last n_post samples.
    """
    wfs = waves[wfin]

    wf_trim = wfs[:, n_pre:n_post]

    if test:
        iwf = 5
        ts = np.arange(wf_trim.shape[1])
        plt.plot(ts, wf_trim[iwf])
        plt.xlabel("clock ticks", ha='right', x=1)
        plt.ylabel("ADC", ha='right', y=1)
        plt.tight_layout()
        plt.show()
        exit()

    return {wfout: wf_trim}


def wavelet():
    """
    placeholder.  this would be pretty cool.
    can the pywavelets module be vectorized?
    can it even be used with np.apply_along_axis?
    """
    print("hi clint")


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


def nlc(waveform, time_constant_samples, fNLCMap, fNLCMap2=None, n_bl=100):
    """
    i guess ben had this working for some radford nonlinearity files,
    maybe they're in GAT or on NERSC somewhere.
    this just looks like a direct port of
    https://github.com/mppmu/MGDO/blob/master/Transforms/MGWFNonLinearityCorrector.cc
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
                    # maybe needs to be +=! check
                    waveform[i] -= fNLCMap2[wf_pt_int + map_offset] + current_inl

        except IndexError:
            print("\nadc value {} and int {}".format(waveform[i], wf_pt_int))
            print("wf_offset {}".format(map_offset))
            print("looking for index {}/{}"
                  .format(wf_pt_int + map_offset, len(fNLCMap)))

    return waveform


def trap_test(waves,
              calcs,
              rise,
              flat,
              decay=0,
              fall=None,
              wfin="wf_blsub",
              wfout="wf_trap",
              test=False):
    """
    compare a few different trapezoid calculations, on single wfs
    inputs are in Hz (clk) and microseconds (rise, flat, decay)
    """
    wfs = waves[wfin]
    clk = waves["settings"]["clk"] # Hz
    nwfs, nsamp = wfs.shape[0], wfs.shape[1]

    # rise, flat, fall = 1, 1.5, 1 # fixed-time-pickoff for energy trapezoid
    rise, flat, fall = 4, 2.5, 4  # energy trapezoid
    # rise, flat, fall = 4, 2, 4 # even dims
    # rise, flat, fall = 3, 2, 1 # asymmetric short-fall (less useful)
    # rise, flat, fall = 0.04, 0.1, 2 # asymmetric short-rise (t0 trigger)

    # convert params to units of [num samples, i.e. clock ticks]
    nsamp = 1e10 / clk
    rt, ft, dt = int(rise * nsamp), int(flat * nsamp), decay * nsamp
    flt = rt if fall is None else int(fall * nsamp)

    # convert params to units of [num samples, i.e. clock ticks]
    nsamp = int(1e10 / clk)
    rt, ft, dt = int(rise * nsamp), int(flat * nsamp), decay * nsamp
    flt = rt if fall is None else int(fall * nsamp)

    # try a few different methods of convolving with
    # a trapezoidal (step-like) kernel

    t1 = time.time()
    kernel = np.zeros(rt + ft + flt)
    kernel[:rt] = 1 / rt
    kernel[rt + ft:] = -1 / flt
    atrap = ndimage.convolve1d(wfs, kernel, axis=1)  # slow?
    print("ndimage.convolve1d: {:.3f}".format(time.time() - t1))

    t2 = time.time()
    atrap = np.zeros_like(wfs)
    for i, wf in enumerate(wfs):
        atrap[i, :] = np.convolve(wf, kernel, 'same')

    # atrap = np.array([np.convolve(wf, kernel, 'same') for wf in wfs])
    # print(atrap.shape)

    npad = rt + int(ft / 2)
    atrap = np.pad(atrap, ((0, 0), (npad, 0)), mode='constant')[:, :-npad]
    print("np.convolve: {:.3f}".format(time.time() - t2))

    t3 = time.time()
    kernel = np.zeros(len(wfs[0]))
    kernel[:rt] = 1 / rt
    kernel[rt + ft:] = -1 / flt
    kernel = np.array(wfs.shape[0] * (kernel, ))
    # atrap = signal.convolve(wfs, kernel, 'same')
    atrap = signal.fftconvolve(wfs, kernel, 'same', axes=1)
    print("convolve elapsed: {:.3f}".format(time.time() - t1))

    t4 = time.time()
    # check against the cumsum method.
    # NOTE: this is faster, but doesn't work for the asymmetric trapezoid

    # TODO (someday): change this method to be recursive
    # https://www.sciencedirect.com/science/article/pii/0168900294910111

    tr1, tr2, tr3 = np.zeros_like(wfs), np.zeros_like(wfs), np.zeros_like(wfs)
    tr1[:, rt:] = wfs[:, :-rt]
    tr2[:, (ft + rt):] = wfs[:, :-rt - ft]
    tr3[:, (rt + ft + flt):] = wfs[:, :-rt - ft - flt]
    scratch = (wfs - tr1) - (tr2 - tr3)
    ctrap = np.cumsum(scratch, axis=1) / rt
    print("cumsum elapsed: {:.3f}".format(time.time() - t4))

    # # pole-zero correct the trapezoid
    # if dt != 0:
    #     rc = 1 / np.exp(1 / dt)
    #     num, den = [1, -1], [1, -rc]
    #     ptrap = signal.lfilter(den, num, atrap)

    if test:

        iwf = 2
        while True:
            if iwf != 2:
                inp = input()
                if inp == "q": exit()
                if inp == "p": iwf -= 2
                if inp.isdigit(): iwf = int(inp) - 1
            iwf += 1
            print(iwf)
            plt.cla()

            wf, ts = wfs[iwf], np.arange(len(wfs[iwf]))
            plt.plot(ts, wf, '-b', alpha=0.5, label='raw wf')

            # plt.plot(ts, scratch[iwf], '-k', label='scratch')
            # plt.plot(ts, trap_wfs[iwf], '-g', lw=3, label='trap')
            # plt.plot(ts, pz_wfs[iwf], '-b', label='pz_trap, {}'.format(dt))

            # check against ben's function
            # import pygama.sandbox.base_transforms as pt
            # trapwf = pt.trap_filter(wfs[iwf], 400, 250, 7200)
            # plt.plot(np.arange(len(trapwf)), trapwf, '-m', label='bentrap')

            # -- simple loop-based trap
            # reproduces the algorithm on p. 74 of clint's thesis.
            # it works for symmetric traps, but is wrong for asym traps
            looptrap = np.zeros(len(wf))
            r1vals, r2vals = [], []
            for i in range(len(wf) - (rt + ft + flt)):

                # sum samples 0 --> rt
                r1 = np.sum(wf[i:rt + i]) / rt
                # r1 = np.sum( wf[i : rt + i] )
                r1vals.append(r1)

                # sum samples rt+ft --> rt+ft+flt
                r2 = np.sum(wf[(rt + ft) + i:(rt + ft + flt) + i]) / flt
                # r2 = np.sum( wf[(rt+ft) + i : (rt+ft+flt) + i] )
                r2vals.append(r2)

                looptrap[i + (rt + ft + flt)] = r2 - r1

            r1vals, r2vals = np.array(r1vals), np.array(r2vals)
            # plt.plot(np.arange(len(r1vals)) + rt, r1vals, '-r', lw=4, label='r1vals')
            # plt.plot(np.arange(len(r2vals)) + rt + ft, r2vals, '-m', lw=4, label='r2vals')
            plt.plot(ts, looptrap, '-k', lw=2, label="loop trap")

            # -- asym trap, method 2, with cumsums
            # atr1, atr2, atr3 = np.zeros(nwf), np.zeros(nwf), np.zeros(nwf)
            # atr1[rt:] = wf[:nsamp-rt]
            # atr2[rt+ft:] = wf[:nsamp-rt-ft]
            # atr3[rt+ft+flt:] = wf[:nsamp-rt-ft-flt]
            # tmp1 = np.cumsum(wf-atr1 - 2*atr2 + 2*atr3)
            # tmp2 = np.cumsum(wf-atr1) / rt
            # tmp3 = np.cumsum(atr2 + atr3)
            # plt.plot(ts, tmp1, c='orange', label="w1")
            # plt.plot(ts, tmp2, c='blue', label="w2")
            # plt.plot(ts, np.cumsum(wf-atr1-2*atr2+2*atr3)/flt, c='cyan', label="w3")
            #
            # NOTE: the lin combinations of the cumsums
            # seem to reproduce the shape of the trap, but they're much more
            # susceptible to noise since they don't look like they use
            # the moving window's average to smooth the output.

            # -- WINNER! --
            # -- asym trap, method 3, trying out np.convolve,
            # with a trapezoidal (step-like) kernel.
            # kernel = np.zeros(rt+ft+flt)
            # kernel[:rt] = 1 / rt
            # kernel[rt+ft:] = -1 / flt
            # atrap = np.convolve(wf, kernel, 'valid')
            # atrap = np.pad(atrap, (rt+ft+flt-1,0), mode='constant')
            # print(atrap.shape, wf.shape)
            # ---> perfect. moved this above to operate on the whole wf block.

            plt.plot(ts, atrap[iwf], '-r', label='atrap')

            # plot max point of trap on the wf (useful for trigger point stuff)
            itrig = np.argmax(atrap[iwf])
            plt.plot(ts[itrig], wf[itrig], '.m', ms=10, label="trig pt")

            plt.xlabel("clock ticks", ha='right', x=1)
            plt.ylabel("ADC", ha='right', y=1)
            plt.legend()
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.001)

    exit()


def peakdet_test(waves, calcs, delta, sigma, ihi, test=False):
    """
    do a speed test of the two peakdet methods
    """
    import time
    from pygama.utils import peakdet as eli_peakdet

    start = time.time()
    print("sigma is", sigma)
    tmp = peakdet(waves, calcs, delta, ihi, sigma)
    tmp1 = tmp["wf_maxc"]
    print(
        "vectorized took {:.4f} sec.  tmp1 shape:".format(time.time() - start),
        tmp1.shape)

    start = time.time()
    wfc = waves["wf_current"]
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
