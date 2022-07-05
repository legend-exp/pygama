import pandas as pd
import matplotlib.pyplot as plt
#plt.style.use('../../pygama/clint.mpl')
import numpy as np
import json
import sys, os
from transforms import *


def main(run):

    #Load runDB File for transform reference
    with open('runDB.json') as f:
        runDB = json.load(f)
    #for item in runDB:
    #    print(item)

    #Separate signal processing & calculations
    raw_to_dsp = runDB['build_options']['conf1']['raw_to_dsp_options']

    #Pull a waveform for testing
    df = pd.read_hdf("../../../data/coherent/tier1/t1_run1796.h5", key='ORSIS3316WaveformDecoder')
    wf = df.iloc[3089][8:]

    A = []
    for i in range(0, 200):
        try:
            wf = df.iloc[i][8:]
            bl = blsub(wf)
            curr = current(bl)
            maximum, minimum = np.amax(curr), np.amin(curr)
            if maximum > np.abs(minimum):
                A.append(maximum)
            else:
                A.append(minimum)
            plt.figure(1)
            plt.plot(curr)
        except:
            continue
    print(A)
    plt.figure(0)
    plt.hist(A, bins=30, histtype='step')
    plt.figure(1)
    plt.show()




if __name__ == '__main__':
    main(sys.argv[1])