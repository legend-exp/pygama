import numpy as np
import matplotlib.pyplot as plt

def avg_baseline(waves, calcs, i_start=0, i_end=500):
    """
    Simple mean, vectorized version of baseline calculator
    """
    wf_block = waves["waveform"]

    # find wf means
    avgs = np.mean(wf_block[:, i_start:i_end], axis=1)

    # add the result as a new column
    calcs["bl_avg"] = avgs
    return calcs


def fit_baseline(waves, calcs, i_start=0, i_end=500, order=1, test=False):
    """
    Polynomial fit [order], vectorized version of baseline calculator
    TODO: arbitrary orders?
    """
    wf_block = waves["waveform"]

    nsamp = wf_block.shape[1]
    if i_end > nsamp:
        i_end = nsamp-1

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
        fit = lambda t : m * t + b
        plt.plot(blts, fit(blts), c='k', lw=3,
                 label='baseline, pol1: \n{:.2e}*ts + {:.1f}'.format(m, b))

        plt.xlim(0, 1100)
        plt.xlabel("clock ticks", ha='right', x=1)
        plt.ylabel("ADC", ha='right', y=1)
        plt.legend(loc=2)
        plt.tight_layout()
        plt.show()
        exit()

    # add the result as new columns
    calcs["bl_int"] = pol[:,0]
    calcs["bl_slope"] = pol[:,1]


def trap_max(waves, calcs, test=False):
    """
    calculate maximum of trapezoid filter
    """
    traps = waves["wf_trap"]

    maxes = np.amax(traps, axis=1)

    if test:
        iwf = 5
        ts = np.arange(len(traps[iwf]))
        plt.plot(ts, waves["wf_blsub"][iwf], '-b', alpha=0.7, label='raw wf')
        plt.plot(ts, traps[iwf], '-k', label="pz corrected trap")
        imaxes = np.argmax(traps, axis=1)
        plt.plot(ts[imaxes[iwf]], maxes[iwf], ".m", markersize=20, label="max")
        plt.xlabel("clock ticks", ha='right', x=1)
        plt.ylabel("ADC", ha='right', y=1)
        plt.legend()
        plt.tight_layout()
        plt.show()
        exit()

    calcs["trap_max"] = maxes


def current_max(waves, calcs, sigma=1, test=False):
    """
    finds the maximum current ("A").
    """
    wfc = waves["wf_current"]

    amax = np.amax(wfc, axis=1) # lol, so simple

    if test:
        import matplotlib.pyplot as plt
        from pygama.utils import peakdet

        wfs = waves["wf_blsub"]

        iwf = -1
        f = plt.figure()
        while True:
            if iwf != -1 and input()=="q": exit()
            iwf += 1
            print(iwf)

            ts = np.arange(len(wfs[iwf]))
            wf = wfs[iwf] / np.amax(wfs[iwf])
            awf = wfc[iwf] / np.amax(wfc[iwf])
            aval = amax[iwf]

            # try out eli's pileup detector
            maxpks, minpks = peakdet(wfc[iwf], 1)
            if len(maxpks > 0):
                print(maxpks[:,0]) # indices
                print(maxpks[:,1]) # found values

            # real crude zoom into the rising edge
            t50 = np.argmax(awf)
            tlo, thi = t50-300, t50+300
            ts, wf, awf = ts[tlo:thi], wf[tlo:thi], awf[tlo:thi]

            plt.cla()
            plt.plot(ts, wf, '-b', alpha=0.7, label='data')
            plt.plot(ts, awf, '-k', label='current')

            for pk in maxpks:
                if tlo < pk[0] < thi:
                    plt.plot(pk[0], pk[1] / np.amax(wfc[iwf]), '.m', ms=20)

            plt.xlabel("clock ticks", ha='right', x=1)
            plt.ylabel('ADC', ha='right', y=1)
            plt.legend()
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.01)

    calcs["current_max"] = amax


def wf_peakdet(waves, calcs, test=False):
    """
    find peaks in the current signal (multisite events and pileup)
    """
    print("hi clint")


def tail_fit(waves, calcs, test=False):
    """
    look around for an exp version of np.polyfit, use it to fit the tail.
    would probably need a separate calculator to know where the tail starts.
    """
    print("hi clint")