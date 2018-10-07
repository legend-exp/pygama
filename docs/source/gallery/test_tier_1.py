"""test_tier_1 docstring, as required by sphinx docs"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt

from pygama.processing import process_tier_0, process_tier_1, TierOneProcessorList
import pygama.decoders as dl
from pygama.transforms import *
from pygama.calculators import *

def main():
    runNumber = 35366
    # runNumber = 11510
    n_max = np.inf

    process(runNumber, n_max=n_max)
    plot_spectrum(runNumber)

def plot_spectrum(runNumber):
    file_name = "t2_run{}.h5".format(runNumber)
    df =  pd.read_hdf(file_name, key="data")

    #show everything in the data frame
    print(df.head())

    #make a hist of energy
    df.hist("trap_max", bins=100)
    plt.show()

def process(runNumber, n_max=5000):
    file_name = "t1_run{}.h5"


    proc_list = get_processing_list()

    runList = [runNumber]
    process_tier_1("", runList, proc_list)

def get_processing_list():

    procs = TierOneProcessorList()

    #pass channel thru to t1
    procs.AddFromTier0("channel")

    #baseline remove
    procs.AddCalculator(fit_baseline, {"end_index":700}, output_name=["bl_slope", "bl_int"])
    procs.AddTransform(remove_baseline, {"bl_0":"bl_int", "bl_1":"bl_slope"}, output_waveform="blrm_wf")

    #energy estimator: pz correct, calc trap
    procs.AddTransform(pz_correct, {"rc":72}, input_waveform="blrm_wf", output_waveform="pz_wf")
    procs.AddTransform(trap_filter, {"rampTime":400, "flatTime":200}, input_waveform="pz_wf", output_waveform="trap_wf")

    procs.AddCalculator(trap_max, {}, input_waveform="trap_wf", output_name="trap_max")
    procs.AddCalculator(trap_max, {"method":"fixed_time","pickoff_sample":400}, input_waveform="trap_wf", output_name="trap_ft")

    return procs

if __name__=="__main__":
    main()
