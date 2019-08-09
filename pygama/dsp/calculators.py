import time
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import scipy.signal as signal

# silence harmless warnings
import warnings
warnings.filterwarnings(action="ignore", module="numpy.ma", category=np.RankWarning)
warnings.filterwarnings(action="ignore", category=RuntimeWarning)


def avg_bl(waves, calcs, ilo=0, ihi=500, wfin="waveform", calc="bl_p0", test=False):
    """
    simple mean, vectorized baseline calculator
    """
    wfs = waves["waveform"]

    # find wf means
    avgs = np.mean(wfs[:, ilo:ihi], axis=1)

    # add the result as a new column
    calcs[calc] = avgs

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
            plt.xlabel("clock ticks", ha='right', x=1)
            plt.ylabel("ADC", ha='right', y=1)
            plt.legend()
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.001)


def dADC(waves, calcs, ilo_1=0, ihi_1=500, ilo_2=1499, ihi_2=3000, wfin="waveform", calc="dADC", test=False):
    """
    subtracting average inside a window on baseline from average inside window on tail
    """
    wfs = waves["waveform"]

    # find wf means
    dADC = np.mean(wfs[:, ilo_2:ihi_2], axis=1) - np.mean(wfs[:, ilo_1:ihi_1], axis=1)

    # add the result as a new column
    calcs[calc] = dADC

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
            plt.xlabel("clock ticks", ha='right', x=1)
            plt.ylabel("ADC", ha='right', y=1)
            plt.legend()
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.001)


def fit_bl(waves, calcs, ilo=0, ihi=500, order=1, wfin="waveform", test=False):
    """
    baseline calculator, uses np.polyfit
    discussed on a Feb 2019 legend S/A call that using a 2nd order term
    in the baseline might be useful in high event-rate situations where the
    baseline hasn't yet fully recovered to flat.  it's also good to reject noise
    """
    wfs = waves[wfin]

    # grab baselines
    x = np.arange(ilo, ihi)
    wfbl = wfs[:, ilo:ihi]

    # run polyfit
    pol = np.polyfit(x, wfbl.T, order).T
    pol = np.flip(pol, 1) # col0:p0, col1:p1, col2:p2, etc.
    wfstd = np.std(wfbl, axis=1) # get the rms noise too

    # save results
    calcs["bl_rms"] = wfstd
    for i, col in enumerate(pol.T):
        calcs["bl_p{}".format(i)] = col

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
            ts = np.arange(wfs[iwf].shape[0])
            plt.plot(ts, wfs[iwf], '-k', label=wfin)

            blwf, blts = wfbl[iwf], np.arange(len(wfbl[iwf]))
            plt.plot(blts, blwf, '-r')

            b, m = pol[iwf]
            fit = lambda t: m * t + b
            plt.plot(blts, fit(blts), c='k', lw=3,
                     label='baseline, pol1: \n{:.2e}*ts + {:.1f}'.format(m, b))

            plt.xlim(0, 1100)
            plt.xlabel("clock ticks", ha='right', x=1)
            plt.ylabel('ADC', ha='right', y=1)
            plt.legend(loc=2)
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.01)


def get_max(waves, calcs, wfin="wf_trap", calc="trap_max", test=False):
    """
    calculate maxima of each row of a waveform block (e.g. a trap filter).
    note that this is very general and works w/ any wf type.
    creates two columns:  max value, and index of maximum.
    """
    wfs = waves[wfin]
    clk = waves["settings"]["clk"]  # Hz

    maxes = np.amax(wfs, axis=1)
    imaxes = np.argmax(wfs, axis=1)
    
    cname = wfin.split("_")[-1]
    calcs["{}_max".format(cname)] = maxes
    calcs["{}_imax".format(cname)] = imaxes

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
            plt.plot(ts, wfs[iwf], '-k', label=wfin)

            raw_wf = waves["wf_blsub"][iwf]
            raw_wf *= np.amax(wf) / np.amax(raw_wf)
            ts = np.arange(len(wf))

            plt.plot(ts, raw_wf, '-b', alpha=0.7, label="raw_wf, normd")
            plt.plot(ts, wf, "-k", label=wfin)
            plt.plot(ts[imaxes[iwf]], maxes[iwf], ".m", ms=20, label="max")

            plt.xlabel("clock ticks", ha='right', x=1)
            plt.ylabel('ADC', ha='right', y=1)
            plt.legend(loc=2)
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.01)


def timepoint(waves, calcs, pct, wfin="wf_savgol", calc="tp", test=False):
    """
    for an estimate of where the wf tail starts, just use pct = 100 + (delta).
    """
    wfs = waves[wfin]
    max = wfin.split('_')[-1] + "_max"
    smax = calcs[max].values

    for p in pct:
        tp_idx = np.argmax(wfs >= smax[:, None] * (p / 100.), axis=1)
        calcs["tp{}".format(p)] = tp_idx

    if test:
        wfraw = waves["wf_blsub"]
        iwf = -1
        while True:
            if iwf != -1:
                inp = input()
                if inp == "q": exit()
                if inp == "p": iwf -= 2
            iwf += 1
            print(iwf)

            wf = wfs[iwf]
            ts = np.arange(len(wf))

            plt.cla()
            plt.plot(ts, wfraw[iwf], "-b", alpha=0.6, label='raw wf')
            plt.plot(ts, wf, "-k", label=wfin)

            cmap = plt.cm.get_cmap('jet', len(pct) + 1)
            for i, tp in enumerate(pct):

                idx = calcs["tp{}".format(tp)].iloc[iwf]
                print("tp{}: idx {}  val {:.2f}".format(tp, idx, wf[idx]))

                plt.plot(idx, wf[idx], ".", c=cmap(i), ms=20,
                         label="tp{}".format(tp))

            plt.xlabel("clock ticks", ha='right', x=1)
            plt.ylabel("ADC", ha='right', y=1)
            plt.legend()
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.01)


def ftp(waves, calcs, wf1="wf_etrap", wf2="wf_atrap", test=False):
    """
    Jason says the fixed time pickoff for MJD ends up being 2 us into the
    2.5 us trap, and the choice is not super important.

    Ian says the trap flat top needs to be as long as a typical rising edge,
    should verify that 2.5 us is good enough for MJ60

    It looks like the asym trap (0.04-0.1-2) is much better at finding
    the t0 time than the short trap (1-1.5-1).  And, by padding it half the
    asym trap's width (in `transforms.trap`), the t0 we find is actually a
    pretty good t0 estimate for the raw waveform as well.
    """
    wflong = waves[wf1]
    wfshort = waves[wf2]

    # get trap settings from metadata
    trap1, trap2 = None, None
    for tr in waves["settings"]["trap"]:
        if tr["wfout"] == wf1: trap1 = tr
        if tr["wfout"] == wf2: trap2 = tr

    # define the fixed time pickoff based on the energy trap settings
    nsamp = 1e10 / waves["settings"]["clk"]
    ftp = int(nsamp * (trap1["rise"] + trap1["flat"]))

    # "walk back" from the short trap's max to get t0.
    # this is less dependent on the trap's baseline noise.
    # Majorana uses a threshold of 2 ADC, hardcoded.
    thresh = 2
    short = wf2.split("_")[-1]
    t0 = np.zeros(wfshort.shape[0], dtype=int)

    # print("WFSHAPE",wfshort.shape, short)
    # print(calcs.columns)
    # print(calcs)
    # exit()

    # damn, i guess i have to loop over the rows
    for i, wf in enumerate(wfshort):
        try:
            imax = calcs[short + "_imax"].iloc[i]
        except:
            print("it happened again!")
            exit()
        trunc = wfshort[i][:imax][::-1]
        t0[i] = len(trunc) - np.where(trunc < thresh)[0][0]

    # save the t0 idx
    calcs['t0'] = t0

    # save the t_ftp idx
    t_ftp = t0 + ftp
    t_ftp[t_ftp >= wflong.shape[1]] = 0  # if t_ftp > len(wf), it failed
    calcs['t_ftp'] = t_ftp

    # save the e_ftp energy
    row_idx = np.arange(wflong.shape[0])
    e_ftp = wflong[np.arange(wflong.shape[0]), t_ftp]
    calcs['e_ftp'] = e_ftp

    if test:

        wfs = waves["wf_blsub"]
        wfsg = waves["wf_savgol"]

        iwf = 2
        while True:
            if iwf != 2:
                inp = input()
                if inp == "q": exit()
                if inp == "p": iwf -= 2
                if inp.isdigit(): iwf = int(inp) - 1
            iwf += 1
            print(iwf)
            wf, ts = wfs[iwf], np.arange(wfs[iwf].shape[0])

            plt.cla()
            plt.plot(ts, wf, '-k', lw=2, alpha=0.5, label='raw wf')
            plt.plot(ts, wfsg[iwf], '-k', lw=1, label='savgol wf')
            plt.plot(ts, wflong[iwf], '-r', label='long: ' + wf1)
            plt.plot(ts, wfshort[iwf], '-b', label='short: ' + wf2)

            smax = calcs[short+"_max"].iloc[iwf]
            simax = calcs[short+"_imax"].iloc[iwf]
            plt.plot(ts[simax], smax, ".k", ms=20, label="short trap max")

            # t0 and t_ftp
            plt.plot(
                ts[t0[iwf]], wfshort[iwf][t0[iwf]], '.g', ms=20, label="t0")
            plt.axvline(
                t_ftp[iwf], c='orange', lw=2, label="t_ftp: {}".format(ftp))

            # e_ftp
            plt.axhline(
                e_ftp[iwf], c='g', label="e_ftp: {:.2f}".format(e_ftp[iwf]))

            plt.xlabel("clock ticks", ha='right', x=1)
            plt.ylabel("ADC", ha='right', y=1)
            plt.legend(loc=2)
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.001)


def num_peaks(waves, calcs, wfin="wf_maxc", test=False):
    """
    take the peakdet wf block and output:
    - the number of maxima
    - the sum of all the maxima
    """
    pks = waves[wfin]

    npeaks = np.count_nonzero(pks, axis=1)
    nsum = np.sum(pks, axis=1)

    calcs["n_curr_pks"] = npeaks
    calcs["s_curr_pks"] = nsum

    if test:
        wfs = waves["wf_notch"]
        wfc = waves["wf_current"]

        iwf = 2
        while True:
            if iwf != 2:
                inp = input()
                if inp == "q": exit()
                if inp == "p": iwf -= 2
                if inp.isdigit(): iwf = int(inp) - 1
            iwf += 1
            print(iwf)
            wf, ts = wfs[iwf], np.arange(wfs[iwf].shape[0])

            plt.cla()
            plt.plot(ts, wf / np.amax(wf), '-k', lw=2, alpha=0.5,
                     label='raw wf')
            plt.plot(ts, wfc[iwf] / np.amax(wfc[iwf]), '-b',
                     label='current wf, {} pks found'.format(npeaks[iwf]))

            ipk = np.where(pks[iwf] > 0)
            for pk in ipk[0]:
                plt.plot(ts[ipk], pks[iwf][ipk] / np.amax(wfc[iwf]), ".m", ms=20)

            plt.xlabel("clock ticks", ha='right', x=1)
            plt.ylabel('ADC', ha='right', y=1)
            plt.legend()
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.01)


def overflow(waves, calcs, wfin="wf_blsub", nbit=14, test=False):
    """
    simple overflow checker.  asks if the max value is at the limit
    of the digitizer's range.  clint had to add a 0.45 factor to get the
    MJ60 wfs to be correctly tagged (ben used 0.5)
    """
    wfs = waves["wf_blsub"]
    maxes = np.amax(wfs, axis=1)
    ovr = maxes > 0.45 * 2**nbit
    calcs["overflow"] = ovr

    if test:
        iwf = 9
        while True:
            if iwf != 9:
                inp = input()
                if inp == "q": exit()
                if inp == "p": iwf -= 2
                if inp.isdigit(): iwf = int(inp) - 1
            iwf += 1
            print(iwf)
            wf, ts = wfs[iwf], np.arange(wfs[iwf].shape[0])

            plt.cla()
            plt.plot(
                ts, wf, '-k', label='raw wf.  overflow? {}'.format(ovr[iwf]))
            plt.xlabel("clock ticks", ha='right', x=1)
            plt.ylabel('ADC', ha='right', y=1)
            plt.legend(loc=4)
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.01)


def tail_fit(waves, calcs, wfin="wf_blsub", delta=1, tp_thresh=0.8, n_check=3,
             order=1, vec=True, test=False):
    """
    this is a "fast" wf fit, not a super duper accurate (slow) one.
    since curve_fit can't be vectorized, this uses np.polyfit.
    we take the log of the wf tail, then fit to a 1st-order pol.
    y(t) = log(A exp(-t/tau)) = log(A)  + (-1/tau) * t
                              = pfit[0] + pfit[1]  * t
    amp = np.exp(pfit[0])
    tau = -1 / pfit[1]
    """
    wfs = waves[wfin]
    ts = np.arange(wfs.shape[1])

    # add a delta to the 100 pct timepoint so we're sure we're on the tail
    nsamp = 1e10 / waves["settings"]["clk"] # Hz
    dt = int(nsamp * delta)
    tp100 = calcs["tp100"] + dt

    # fix out of range timepoints
    tp100[tp100 > tp_thresh * wfs.shape[1]] = 0

    # create a masked array to handle the different-length wf tails
    tails = np.full_like(wfs, np.nan)
    for i, tp in enumerate(tp100):
        tails[i, tp:] = wfs[i, tp:]
    tails = np.ma.masked_invalid(tails)
    log_tails = np.ma.log(tails) # suppress neg value warnings

    t_start = time.time()
    if vec:
        """
        run the vectorized fit, which is faster but sensitive to timepoints
        being too near the end of the wfs -- it throws off the whole matrix.
        so check the fit results against `n_check` random single tail fits.
        """
        pfit = np.ma.polyfit(ts, log_tails.T, 1).T

        amps = np.exp(pfit[:,1])
        taus = -1 / pfit[:,0]
        calcs["tail_amp"] = amps
        calcs["tail_tau"] = taus

        # run a second, higher-order fit to estimate error
        # (i'm lazy and did this instead of returning the covariance matrix)
        pol_fit = np.ma.polyfit(ts, log_tails.T, order).T
        pol_fit = np.flip(pol_fit, axis=1)
        for i, col in enumerate(pol_fit.T):
            calcs["tail_p{}".format(i)] = col

        for iwf in np.random.choice(log_tails.shape[0], n_check):
            check_fit = np.ma.polyfit(ts, log_tails[iwf], order)
            ch_amp = np.exp(check_fit[1])
            ch_tau = -1 / check_fit[0]

            # if within 90%, they're fine. a polyfit mistake is OOM wrong
            pct1 = 100 * (ch_amp - amps[iwf]) / amps[iwf]
            pct2 = 100 * (ch_tau - taus[iwf]) / taus[iwf]
            if (pct1 > 90) | (pct2 > 90):
                print("WARNING: there are probably invalid values in tails.")
                print("iwf {}, check amp: {:.3e}   tau: {:.3e}".format(iwf, ch_amp, ch_tau))
                print("     original amp: {:.3e}   tau: {:.3e}".format(amps[iwf], taus[iwf]))
                print("     amp pct diff: {:.2f}%  tau: {:.2f}%".format(pct1, pct2))
    else:
        """
        run a non-vectorized fit with np.polyfit and np.apply_along_axis.
        for 200 wfs, this is about half as fast as the vectorized mode.
        """
        def poly1d(wf, ts, ord):
            if np.ma.count(wf)==0:
                return np.array([1, 0])
            return np.ma.polyfit(wf, ts, ord)

        if len(log_tails):
          pfit = np.apply_along_axis(poly1d, 1, log_tails, ts, order)

          amps = np.exp(pfit[:,1])
          taus = -1 / pfit[:,0]
          calcs["tail_amp"] = amps
          calcs["tail_tau"] = taus

    # print("Done.  Elapsed: {:.2e} sec.".format(time.time()-t_start))
    # exit()

    if test:
        wfbl = waves["wf_blsub"]
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
            plt.plot(ts, wfs[iwf], '-k', label=wfin)
            plt.plot(ts, wfbl[iwf], '-b', alpha=0.4, label="wf_blsub")

            # get the wf tail
            wf_tail = np.ma.filled(tails[iwf,:], fill_value = np.nan)
            idx = np.where(~np.isnan(wf_tail))
            wf_tail, ts_tail = wf_tail[idx], ts[idx]
            plt.plot(ts_tail, wf_tail, '-g', label='tail')

            # show the np.polyfit result
            amp, tau = amps[iwf], taus[iwf]
            plt.plot(ts_tail, amp * np.exp(-ts_tail/tau), '-r',
                     label="polyfit dc: {:.1f}".format(tau/100))

            # compare against curve_fit, with exponential. (not easily vectorized)
            from scipy.optimize import curve_fit
            tmax = np.amax(wf_tail)
            def gaus(t, a, tau):
                return a * np.exp(-t/tau)
            pars, pcov = curve_fit(gaus, ts_tail, wf_tail,
                                   p0=(tmax,8000),
                                   bounds=[[0.8*tmax, 5000],[1.2*tmax, 20000]])
            perr = np.sqrt(np.diag(pcov))
            dc, dc_err = pars[1] / 100, perr[1] / 100
            plt.plot(ts_tail, gaus(ts_tail, *pars), '-m', lw=3,
                     label="curve_fit dc: {:.1f} +/- {:.3f}".format(dc, dc_err))

            plt.xlabel("clock ticks", ha='right', x=1)
            plt.ylabel('ADC', ha='right', y=1)
            plt.legend(loc=4)
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.01)


def dcr(waves, calcs, delta=1, t_win2=25, wlen=2, tp_thresh=0.8, wfin="wf_savgol", test=False):
    """
    calculate the slope of the wf tail from taking the average of two windows.
    first one is a (tp100+delta), the second one is (t_win2).
    (delta, t_win2, wlen) are in us.

    TODO:
    - try "true" pole zero correction: (applying de-convolution of the full
      channel-specific electronic response function before searching for a
      remaining slow component. i.e. use `wf_pz` as the input)
    - charge trapping correction, v1 (ftp vs trap_max energy params, this
      is the "effective pole zero" described in the DCR unidoc)
    - charge trapping correction, v2 (use drift time (tp20-t0) to calculate
      the expected amount of charge lost in the bulk.)
    - could try fitting each window to a line with np.polyfit, then comparing
      the two slopes
    - add a "mode" option which selects the various improvements above.
    """
    wfs = waves[wfin]
    ts = np.arange(wfs.shape[1])

    # add a delta to the 100 pct timepoint so we're sure we're on the tail
    nsamp = 1e10 / waves["settings"]["clk"] # Hz
    win = int(nsamp * wlen)
    dt = int(nsamp * delta)
    tp100 = calcs["tp100"] + dt
    iwin2 = int(nsamp * t_win2)

    # fix out of range timepoints
    tp100[tp100 > tp_thresh * wfs.shape[1]] = 0

    # compute average in window 1 (use masked arrays)
    win_1 = np.full_like(wfs, np.nan)
    for i, ilo in enumerate(tp100):
        win_1[i, ilo:ilo+win] = wfs[i, ilo:ilo+win]
    win_1 = np.ma.masked_invalid(win_1)
    avg_1 = np.sum(win_1, axis=1) / np.count_nonzero(win_1, axis=1)

    # compute average in window 2  (always fixed)
    win_2 = np.full_like(wfs, np.nan)
    win_2[:, iwin2:iwin2+win] = wfs[:, iwin2:iwin2+win]
    win_2 = np.ma.masked_invalid(win_2)
    avg_2 = np.sum(win_2, axis=1) / np.count_nonzero(win_2, axis=1)

    # get two-point tail slope
    # sig = (y1 - y2) / (t1 - t2) # pg 4, dcr unidoc
    y1, y2 = avg_1, avg_2
    t1 = (tp100.values) + win/2
    t2 = (iwin2 + win/2) * np.ones_like(tp100.values)
    num, den = y1 - y2, t1 - t2
    slope = np.divide(num, den)

    # # apply charge trapping correction ("v1", pg. 12 of DCR unidoc).
    # relies on input parameters A and \lambda.  Maybe better to do this
    # in Tier 2 processing, or skip straight to the v2 algorithm
    # e_max, e_ftp = calcs["etrap_max"], calcs["e_ftp"]

    # # apply charge trapping correction ("v2")
    # t0, t20 = calcs["t0"], calcs["tp50"]
    # drift_time = t20 - t0

    # name the output parameter based on the input wf name
    wf_type = wfin.split("_")[-1]
    calcs["tslope_{}".format(wf_type)] = slope

    if test:
        wfbl = waves["wf_blsub"]
        # from pygama.utils import set_plot_style
        # set_plot_style("clint")
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
            plt.plot(ts, wfs[iwf], '-k', label=wfin)
            plt.plot(ts, wfbl[iwf], '-b', alpha=0.4, label="wf_blsub")

            idx1 = np.where(~np.isnan(win_1[iwf]))
            idx2 = np.where(~np.isnan(win_2[iwf]))

            plt.plot(ts[idx1], win_1[iwf][idx1], '-r', lw=10, alpha=0.5)
            plt.plot(ts[idx2], win_2[iwf][idx2], '-r', lw=10, alpha=0.5)

            slo = (avg_1[iwf] - avg_2[iwf]) / (t1[iwf] - t2[iwf])
            xv = np.arange(t1[iwf], t2[iwf])
            yv = slo * (xv - t1[iwf]) + avg_1[iwf]

            plt.plot(xv, yv, '-r', label="slope: {:.2e}".format(slo))
            # plt.plot(np.nan, np.nan, ".w", label="main: {:.2e}".format(slope[iwf]))

            plt.plot(t1[iwf], avg_1[iwf], ".g", ms=20)
            plt.plot(t2[iwf], avg_2[iwf], ".g", ms=20)

            plt.xlabel("Clock ticks", ha='right', x=1)
            plt.ylabel('ADC', ha='right', y=1)
            plt.legend(loc=4)
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.01)


def drift_time(waves, calcs, test=False):
    """
    do the tp[pct] - t0 time.
    could maybe also try a np polyfit to roughly
    estimate the curvature? idk, maybe simpler is better
    """
    print("hi clint")


def gretina_overshoot(rc_us, pole_rel, freq=100E6):
    """
    for use with scipy.signal.lfilter
    """
    zmag = np.exp(-1. / freq / (rc_us * 1E-6))
    pmag = zmag - 10.**pole_rel
    num = [1, -zmag]
    den = [1, -pmag]
    return (num, den)


def fir():
    """
    FIR Filter, fir the win ;-)
    https://docs.scipy.org/doc/scipy-1.2.1/reference/generated/scipy.signal.firwin.html

    This might be better than computing a whole bunch of notch filters.
    Just do a study on the MJ60 power spectrum, and create a multiband filter

    FIR FAQ
    https://dspguru.com/dsp/faqs/fir/basics/
    """
    print("hi clint")
    numtaps = 3
    f = 0.1
    signal.firwin(numtaps, f)


def cfit():
    """
    a curve_fit (or optimize.minimize, or lmfit)
    apply_along_axis function might be good, for special cases
    when we don't care about using more computation time
    """
    # # curve_fit, with exponential. (not easily vectorized)
    # from scipy.optimize import curve_fit
    # tmax = np.amax(wf_tail)
    # def gaus(t, a, tau):
    #     return a * np.exp(-t/tau)
    # pars, pcov = curve_fit(gaus, ts_tail, wf_tail,
    #                        p0=(tmax,8000),
    #                        bounds=[[0.8*tmax, 5000],[1.2*tmax, 20000]])
    # perr = np.sqrt(np.diag(pcov))
    # dc, dc_err = pars[1] / 100, perr[1] / 100
    # plt.plot(ts_tail, gaus(ts_tail, *pars), '-m', lw=3,
    #          label="curve_fit dc: {:.1f} +/- {:.3f}".format(dc, dc_err))
    print("hi clint")
