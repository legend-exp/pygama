import numpy as np

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
    """
    calculate maximum of trapezoid filter - no pride here
    """
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


def current_max(waves, calcs, sigma=1, test=False):
    """
    finds the maximum current ("A").
    """
    wfc = waves["wf_current"]

    amax = np.amax(wfc, axis=1)

    if test:
        import matplotlib.pyplot as plt
        from pygama.utils import peakdet

        wfs = waves["wf_blsub"]

        # comparison w/ MGDO
        from ROOT import std, MGTWaveform, MGWFTrapSlopeFilter
        tsf = MGWFTrapSlopeFilter()
        tsf.SetPeakingTime(1)
        tsf.SetIntegrationTime(10)
        tsf.SetEvaluateMode(7)

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

            mgwf_in, mgwf_out = MGTWaveform(), MGTWaveform()
            tmp = std.vector("double")(len(wf))
            for i in range(len(wf)): tmp[i] = wf[i]
            mgwf_in.SetData(tmp)
            tmp = mgwf_in.GetVectorData()
            tsf.TransformOutOfPlace(mgwf_in, mgwf_out)
            out = mgwf_out.GetVectorData()
            mgawf = np.fromiter(out, dtype=np.double, count=out.size())
            mgawf = mgawf / np.amax(mgawf)

            # jasondet [12:11 PM]
            # Our A is equivalent to the slope of a linear fit over 10 samples
            # jasondet [11:33 AM]
            # yeah but counting peaks in a current trace is not going to be as
            # robust (esp. for low-E pulses) as thresholding the current trace,
            # that’s what David H-A is working on.
            # TODO: require the sum A be over a particular value

            # try out eli's pileup detector anyway
            maxpks, minpks = peakdet(wfc[iwf], 1)
            if len(maxpks > 0):
                print(maxpks[:,0]) # indices
                print(maxpks[:,1]) # found values

            # real crude zoom into the rising edge
            t50 = np.argmax(awf)
            tlo, thi = t50-300, t50+300
            # ts, wf, awf, mgawf = ts[tlo:thi], wf[tlo:thi], awf[tlo:thi], mgawf[tlo:thi]

            plt.cla()
            plt.plot(ts, wf, '-b', label='data')
            plt.plot(ts, awf, '-r', alpha=0.7, label='pygama')
            # plt.plot(ts, mgawf, '-g', alpha=0.7, label='mgdo')

            for pk in maxpks:
                # if tlo < pk[0] < thi:
                plt.plot(pk[0], pk[1] / np.amax(wfc[iwf]), '.k', ms=10)

            # plt.legend()
            plt.show(block=False)
            plt.pause(0.01)

    calcs["current_max"] = amax


def current_int(waves, calcs, test=False):
    """
    integrate the current signal.
    """
    wfc = waves["wf_current"]

    csum = np.sum(wfc, axis=1)

    if test:
        print(csum.shape)
        exit()

    calcs["current_int"] = csum


def wf_peakdet(waves, calcs, test=False):
    print("hi clint")
