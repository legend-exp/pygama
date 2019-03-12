import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

def avg_bl(waves, calcs, i_start=0, i_end=500, wfin="waveform", calc="bl_avg", test=False):
    """
    simple mean, vectorized baseline calculator
    """
    wf_block = waves["waveform"]

    # find wf means
    avgs = np.mean(wf_block[:, i_start:i_end], axis=1)

    # add the result as a new column
    calcs[calc] = avgs
    return calcs


def fit_bl(waves, calcs, i_start=0, i_end=500, order=1,
           wfin="waveform", cnames=["bl_int", "bl_slope", "bl_rms"], test=False):
    """
    polynomial fit [order], vectorized baseline calculator
    TODO:
    - if we made this calculator a little more general, it could do arb. orders
      on arbitary windows, so it could also be re-used to fit the wf tails.
    - also discussed on a Feb 2019 legend S/A call that using a 2nd order term
      in the baseline might be useful in high event-rate situations where the
      baseline hasn't yet fully recovered to flat.
    """
    wf_block = waves[wfin]

    # run polyfit
    x = np.arange(i_start, i_end)
    wfs = wf_block[:, i_start:i_end].T
    pol = np.polynomial.polynomial.polyfit(x, wfs, order).T

    # get the rms noise
    wfstd = np.std(wfs.T, axis=1)

    if test:
        iwf = 5

        ts, wf = np.arange(len(wf_block[iwf])), wf_block[iwf]
        plt.plot(ts, wf, c='b')

        blwf, blts = wfs.T[iwf], np.arange(len(wfs.T[iwf]))
        plt.plot(blts, blwf, c='r')

        b, m = pol[iwf]
        fit = lambda t: m * t + b
        plt.plot(blts, fit(blts), c='k', lw=3,
                 label='baseline, pol1: \n{:.2e}*ts + {:.1f}'.format(m, b))

        plt.xlim(0, 1100)
        plt.xlabel("clock ticks", ha='right', x=1)
        plt.ylabel("ADC", ha='right', y=1)
        plt.legend(loc=2)
        plt.tight_layout()
        plt.show()
        exit()

    # add the results as new columns
    for i, c in enumerate(["bl_int", "bl_slope"]):
        calcs[c] = pol[:, i]
    calcs["bl_rms"] = wfstd


def get_max(waves, calcs, clk, wfin="wf_trap", calc="trap_max", test=False):
    """
    calculate maxima of each row of a waveform block (e.g. a trap filter).
    note that this is very general and works w/ any wf type.
    creates two columns:  max value, and index of maximum.
    """
    wfs = waves[wfin]

    maxes = np.amax(wfs, axis=1)
    imaxes = np.argmax(wfs, axis=1)

    cname = wfin.split("_")[-1]
    calcs["{}_max".format(cname)] = maxes
    calcs["{}_imax".format(cname)] = imaxes

    if test:
        iwf = 5

        # for reference, show the raw wf, but scale s/t it matches the given wf
        wf = wfs[iwf]
        raw_wf = waves["wf_blsub"][iwf]
        raw_wf *= np.amax(wf) / np.amax(raw_wf)
        ts = np.arange(len(wf))

        plt.plot(ts, raw_wf, '-b', alpha=0.7, label="raw_wf, normd")
        plt.plot(ts, wf, "-k", label=wfin)
        plt.plot(ts[imaxes[iwf]], maxes[iwf], ".m", ms=20, label="max")
        plt.xlabel("clock ticks", ha='right', x=1)
        plt.ylabel("arb", ha='right', y=1)
        plt.legend()
        plt.tight_layout()
        plt.show()
        exit()


def timepoint(waves, calcs, pct, wfin="wf_savgol", calc="tp", test=False):
    """
    for an estimate of where the wf tail starts, just use pct=100.
    """
    wfs = waves[wfin]
    smax = calcs["savgol_max"].values

    for p in pct:
        tp_idx = np.argmax(wfs >= smax[:,None] * (p / 100.), axis=1)
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

            cmap = plt.cm.get_cmap('jet', len(pct)+1)
            for i, tp in enumerate(pct):

                idx = calcs["tp{}".format(tp)][iwf]
                print("tp{}: idx {}  val {:.2f}".format(tp, idx, wf[idx]))

                plt.plot(idx, wf[idx], ".", c=cmap(i), ms=20,
                         label="tp{}".format(tp))

            plt.xlabel("clock ticks", ha='right', x=1)
            plt.ylabel("ADC", ha='right', y=1)
            plt.legend()
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.01)


def mjd_ftp(waves, calcs, test=False):
    """
    dependent on:
    {"wfout":"wf_etrap", "rise":4, "flat":2.5, "decay":72, "clk":100e6},
    {"wfout":"wf_strap", "rise":1, "flat":1.5, "decay":72, "clk":100e6},
    {"wfout":"wf_atrap", "rise":0.04, "flat":0.1, "fall":2, "clk":100e6}

    # todo: need to find the threshold crossing of the short/asym traps,
    # not the maximums.
    """
    print("hi clint")
    exit()


def get_t0(waveform, baseline=0, median_kernel_size=51, max_t0_adc=100):
    """
    max t0 adc: maximum adc (above baseline) the wf can get to,
    before assuming the wf has started
    """
    if np.amax(waveform) < max_t0_adc:
        return np.nan

    wf_med = signal.medfilt(waveform, kernel_size=median_kernel_size)

    med_diff = gaussian_filter1d(wf_med, sigma=1, order=1)

    tp05 = calc_timepoint(waveform, percentage=max_t0_adc, baseline=0,
                          do_interp=False, doNorm=False)

    tp05_rel = np.int(tp05 + 1)
    thresh = 5E-5
    last_under = tp05_rel - np.argmax(med_diff[tp05_rel::-1] <= thresh)
    if last_under >= len(med_diff) - 1:
        last_under = len(med_diff) - 2

    t0 = np.interp(thresh, (med_diff[last_under], med_diff[last_under + 1]),
                   (last_under, last_under + 1))

    return t0


def num_peaks():
    """
    placeholder.  grab the peakdet wf block and just output the number of
    maxima.  this is single valued, at least.
    """
    print("hi clint")


def overflow(waves, calcs, nbit=14, test=False):
    """
    need to test on some saturated wfs from mj60
    """
    # non-vectorized version
    # return True if np.amax(waveform) >= 0.5 * 2**nbit - 1 else False
    print("hi clint")


def tail_fit(waves, calcs, test=False):
    """
    let's use the 100% timepoint to set where the tail starts.

    to start, i looked around for an exp version of np.polyfit
    this is a good starting point - reduce to something that can be
    done using polyfit, since "np.expofit" doesn't exist
    https://stackoverflow.com/questions/3433486/how-to-do-exponential-and-logarithmic-curve-fitting-in-python-i-found-only-poly
    """
    print("hi clint")


def gretina_overshoot(rc_us, pole_rel, freq=100E6):
    """
    """
    zmag = np.exp(-1. / freq / (rc_us * 1E-6))
    pmag = zmag - 10.**pole_rel

    num = [1, -zmag]
    den = [1, -pmag]

    return (num, den)
