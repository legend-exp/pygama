import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt

def main():

    # slice data (delete this part)

    # load data
    # waves, calcs =

    # run cfd calculator
    cfd(waves, calcs)


def cfd(waves, calcs, frac=0.1, delay=2, win=0.05, wfin="wf_blsub", test=False):
    """
    in a constant fraction discriminator, the signal is split into two parts.
    traditionally, one part is time-delayed, and the other is inverted and
    low-pass filtered.

    we use scipy.signal.lfilter, which is very general, and could be used
    to smooth/delay both wfs if we wanted to.

    I kind of think that b/c our signals don't oscillate, the delay is not
    very important, and finding the last crossing is going to be contaminated
    by the baseline RMS noise.
    """
    # convert to clock ticks
    nsamp = 1e10 / waves["settings"]["clk"]
    nd, nwin = int(nsamp * delay), int(nsamp * win)
    print("nd:",nd,"nwin:",nwin)

    wfs = waves[wfin]

    a, b = np.zeros(nd+nwin), np.zeros(nd+nwin)
    # a[0], b[0], b[-1] = 1, -frac, 1 # internet settings
    a[0] = 1
    b[nd:nd+nwin] = -frac

    wf_cfd = signal.lfilter(b, a, wfs, axis=1)

    wfsub = wfs + wf_cfd

    # for i, wf in enumerate(wfs):
    #     # cross_pts =
    #     tol = 1e-5 # tolerance (could use bl rms??)
    #     out = b[(np.abs(a[:,None] - b) < tol).any(0)]
    #     # out = b[np.isclose(a[:,None],b).any(0)]

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
            plt.plot(ts, wf_cfd[iwf], "-r", label="cfd wf")
            plt.plot(ts, wfsub[iwf], "-g", label="sub wf")

            # compare against the ftp

            # plt.ylim(1.2*np.amin(wfs[iwf]), 1.2*np.amax(wfs[iwf]))
            # plt.ylim(bottom=np.amin(wfs[iwf]))
            plt.xlabel("clock ticks", ha='right', x=1)
            plt.ylabel("ADC", ha='right', y=1)
            plt.legend(loc=2)
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.01)




if __name__=="__main__":
    main()