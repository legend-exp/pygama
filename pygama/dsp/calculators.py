import numpy as np
import matplotlib.pyplot as plt


def avg_bl(waves,
           calcs,
           i_start=0,
           i_end=500,
           wfin="waveform",
           cname="bl_avg",
           test=False):
    """
    Simple mean, vectorized baseline calculator
    """
    wf_block = waves["waveform"]

    # find wf means
    avgs = np.mean(wf_block[:, i_start:i_end], axis=1)

    # add the result as a new column
    calcs[cname] = avgs
    return calcs


def fit_bl(waves,
           calcs,
           i_start=0,
           i_end=500,
           order=1,
           wfin="waveform",
           cnames=["bl_int", "bl_slope"],
           test=False):
    """
    Polynomial fit [order], vectorized baseline calculator

    TODO:
    - if we made this calculator a little more general, it could do arb. orders
      on arbitary windows, so it could also be re-used to fit the wf tails.
    - also discussed on a Feb 2019 legend S/A call that using a 2nd order term
      in the baseline might be useful in high event-rate situations where the
      baseline hasn't yet fully recovered to flat.
    """
    wf_block = waves[wfin]

    nsamp = wf_block.shape[1]
    if i_end > nsamp:
        # this could be useful for a tail fit
        i_end = nsamp - 1

    # run polyfit
    x = np.arange(i_start, i_end)
    wfs = wf_block[:, i_start:i_end].T
    pol = np.polynomial.polynomial.polyfit(x, wfs, order).T

    if test:
        iwf = 5

        ts, wf = np.arange(len(wf_block[iwf])), wf_block[iwf]
        plt.plot(ts, wf, c='b')

        blwf, blts = wfs.T[iwf], np.arange(len(wfs.T[iwf]))
        plt.plot(blts, blwf, c='r')

        b, m = pol[iwf]
        fit = lambda t: m * t + b
        plt.plot(
            blts,
            fit(blts),
            c='k',
            lw=3,
            label='baseline, pol1: \n{:.2e}*ts + {:.1f}'.format(m, b))

        plt.xlim(0, 1100)
        plt.xlabel("clock ticks", ha='right', x=1)
        plt.ylabel("ADC", ha='right', y=1)
        plt.legend(loc=2)
        plt.tight_layout()
        plt.show()
        exit()

    # add the results as new columns
    for i, c in enumerate(cnames):
        calcs[c] = pol[:, i]


def get_max(waves, calcs, wfin="wf_trap", cname="trap_max", test=False):
    """
    calculate maxima of each row of a waveform block (e.g. a trap filter).
    note that this is very general and works w/ any wf type.
    """
    wfs = waves[wfin]

    maxes = np.amax(wfs, axis=1)  # lol

    if test:
        iwf = 5
        ts = np.arange(len(wfs[iwf]))
        plt.plot(ts, waves["wf_blsub"][iwf], '-b', alpha=0.7, label='raw wf')
        plt.plot(ts, wfs[iwf], '-k', label="pz corrected trap")
        imaxes = np.argmax(wfs, axis=1)
        plt.plot(ts[imaxes[iwf]], maxes[iwf], ".m", markersize=20, label="max")
        plt.xlabel("clock ticks", ha='right', x=1)
        plt.ylabel("ADC", ha='right', y=1)
        plt.legend()
        plt.tight_layout()
        plt.show()
        exit()

    calcs[cname] = maxes


def get_tmax(waves, calcs, clk, wfin="wf_trap_E", cname="tmax_E", test=False):
    """
    give the t_max of a waveform (in ns).  clk is in Hz
    """
    wfs = waves[wfin]
    wft = 1e9 / clk  # ns

    tmaxes = np.argmax(wfs, axis=1) * wft

    if test:
        iwf = 5
        ts = np.arange(len(wfs[iwf])) * wft
        plt.plot(ts, wfs[iwf], '-b', label="trapE")
        tmp = np.amax(wfs[iwf])
        plt.plot(tmaxes[iwf], tmp, ".m", markersize=20,
            label="tmax:{:.0f} max:{:.1f}".format(tmaxes[iwf], tmp))
        plt.xlabel("time (ns)", ha='right', x=1)
        plt.ylabel("ADC", ha='right', y=1)
        plt.legend(loc=2)
        plt.tight_layout()
        plt.show()
        exit()

    calcs[cname] = tmaxes


def is_saturated(waves, calcs, nbit=14, test=False):
    """
    TODO: need to actually test on some saturated wfs from mj60
    """
    # non-vectorized version
    # return True if np.amax(waveform) >= 0.5 * 2**nbit - 1 else False
    print("hi clint")


def tail_fit(waves, calcs, test=False):
    """
    look around for an exp version of np.polyfit, use it to fit the tail.
    would probably need a separate calculator to know where the tail starts.
    """
    # this is a good starting point - reduce to something that can be
    # done using polyfit.  since "np.expofit" doesn't exist
    # https://stackoverflow.com/questions/3433486/how-to-do-exponential-\
    # and-logarithmic-curve-fitting-in-python-i-found-only-poly
    print("hi clint")
