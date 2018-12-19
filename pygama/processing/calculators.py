import numpy as np
import pandas as pd


def avg_baseline(waves, calcs, i_start=0, i_end=500):
    """ Simple mean, vectorized version of baseline calculator """

    wf_block = waves["waveform"]

    # find wf means
    avgs = np.mean(wf_block[:, i_start:i_end], axis=1)

    # add the result as a new column
    calcs["bl_avg"] = avgs
    return calcs


def fit_baseline(waves, calcs, i_start=0, i_end=500, order=1):
    """ Polynomial fit [order], vectorized version of baseline calculator
    TODO: arbitrary orders?
    """
    wf_block = waves["waveform"]

    if wf_block.size == 0:
        print("Warning, empty block!")
        exit()

    nsamp = wf_block.shape[1]
    if i_end > nsamp:
        i_end = nsamp-1

    # run polyfit
    x = np.arange(i_start, i_end)
    wfs = wf_block[:, i_start:i_end].T
    pol = np.polynomial.polynomial.polyfit(x, wfs, order).T

    # add the result as new columns
    calcs["bl_int"] = pol[:,0]
    calcs["bl_slope"] = pol[:,1]
    return calcs


def trap_max(waves, calcs, test=False):
    """ calculate maximum of trapezoid filter - no pride here """

    wfs = waves["wf_trap"]

    maxes = np.amax(wfs, axis=1)

    if test:
        import matplotlib.pyplot as plt
        iwf = 1
        plt.plot(np.arange(len(wfs[iwf])), wfs[iwf], '-r')
        plt.axhline(maxes[iwf])
        plt.show()
        exit()

    calcs["trap_max"] = maxes
    return calcs
