#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
plt.style.use('./pltReports.mplstyle')

import pygama

import waveLibs as wl

# dataDir = "/Users/wisecg/project/mj60/data"
dataDir = "/Users/wisecg/dev/mj60/data"

def main():

    process_MJ60()
    process_MJD()
    # compareEnergy()


def process_t0(run, nMax, dataDir, verbose=False):
    """ Tier 0 defined in:
    - pygama/processing/processing.py  (public functions)
    - pygama/processing/_pygama.pyx    (main routine)
    """
    runList = [run]
    pygama.process_tier_0(dataDir, runList, output_dir=dataDir, chan_list=None, \
                          n_max=nMax, verbose=True)


def process_t1(run, dataDir):
    """ Tier 1 defined in:
    - pygama/processing/processing.py  (public functions)
    - pygama/processing/_pygama.pyx    (main routine)
    Requires a processor list.
    """
    runList = [run]
    pygama.process_tier_1(dataDir, runList, make_processor_list(), output_dir=dataDir)


def process_MJD():
    """ process a random MJD background run """

    run = 36854
    nMax = 5000

    # process data
    process_t0(run, nMax, dataDir)
    print_tier0_info(run)
    process_t1(run, dataDir)
    return

    # make an energy plot
    dft2 = pd.read_hdf("%s/t2_run%d.h5" % (dataDir, run))
    plt.hist(dft2['trap_max'], bins=300, histtype="step", log=True)
    plt.xlabel("trap_max", ha='right', x=1)
    plt.ylabel("counts", ha='right', y=1)
    plt.savefig("./plots/mjd_energy.pdf")

    # plot a few waveforms
    ft1 = "%s/t1_run%d.h5" % (dataDir, run)
    dc = pygama.Gretina4MDecoder(ft1) # this class reads the "ORGretina4MWaveformDecoder" dataframe
    df_events = pd.read_hdf(ft1, key=dc.decoder_name)

    plt.figure()
    plt.cla()
    nwf = 10
    for i, (index, row) in enumerate(df_events.iterrows()):
        wf = dc.parse_event_data(row)
        # plt.plot(wf.data[0][4:]) # mj60
        plt.plot(wf.data)
        if i > nwf:
            break
    plt.xlabel("Time [ns]", ha='right', x=1)
    plt.ylabel("ADC", ha='right', y=1)
    # plt.show()
    plt.savefig("./plots/mjd_waveforms.pdf")


def process_MJ60():
    # process mj60 run (struck digitizer)

    # run = 72 # bkg data, with buffer wrap
    run = 74 # hv pulser, no buffer wrap mode

    # process data
    process_t0(run, np.inf, dataDir, verbose=True)
    print_tier0_info(run)
    process_t1(run, dataDir)
    return

    # make an energy plot
    dft2 = pd.read_hdf("%s/t2_run%d.h5" % (dataDir, run))
    plt.hist(dft2['trap_max'], bins=300, histtype="step", log=True)
    plt.xlabel("trap_max", ha='right', x=1)
    plt.ylabel("counts", ha='right', y=1)
    plt.savefig("./plots/mj60_energy_run%d.pdf" % run)

    # plot a few waveforms
    ft1 = "%s/t1_run%d.h5" % (dataDir, run)
    dc = pygama.SIS3302Decoder(ft1)
    df_events = pd.read_hdf(ft1, key=dc.decoder_name)

    plt.figure()
    plt.cla()
    nwf = 10
    for i, (index, row) in enumerate(df_events.iterrows()):
        wf = dc.parse_event_data(row)
        # print(len(wf.data))
        plt.plot(wf.data)
        if i > nwf:
            break
    plt.xlabel("Time [ns]", ha='right', x=1)
    plt.ylabel("ADC", ha='right', y=1)
    # plt.show()
    plt.savefig("./plots/mj60_waveforms_run%d.png" % run)


def print_tier0_info(run):
    """ The pygama Tier 1 HDF5 contains multiple datasets,
    whose keys must be specified when we do pd.read_hdf(.., key="name")
    Tier 2 doesn't have this structure, so the key is not needed.
    """
    print("\n\nDone with processing .... Printing tier info .... run",run)

    # get a list of keys
    print("Available Tier 1 keys:")
    store = pd.HDFStore("%s/t1_run%d.h5" % (dataDir, run))
    for g in store.groups():
        print("  ", g)
    store.close()

    # MJ60
    if run == 74:

        dft1_sis = pd.read_hdf("{}/t1_run{}.h5".format(dataDir, run), key="ORSIS3302DecoderForEnergy")
        dft1_mod = pd.read_hdf("{}/t1_run{}.h5".format(dataDir, run), key="ORSIS3302Model")

        print(dft1_sis.columns)
        print(dft1_mod.columns)

        print("Found {} ORSIS3302DecoderForEnergy events".format(len(dft1_sis)))
        print("Found {} ORSIS3302Model events".format(len(dft1_mod)))

    # MJD
    if run == 36854:

        dft1_gret = pd.read_hdf("%s/t1_run%d.h5" % (dataDir, run), key="ORGretina4MModel")
        dft1_wave = pd.read_hdf("%s/t1_run%d.h5" % (dataDir, run), key="ORGretina4MWaveformDecoder")
        dft1_pre = pd.read_hdf("%s/t1_run%d.h5" % (dataDir, run), key="ORMJDPreAmpDecoderForAdc")
        dft1_iseg = pd.read_hdf("%s/t1_run%d.h5" % (dataDir, run), key="ORiSegHVCardDecoderForHV")

        print(dft1_gret.columns)
        print(dft1_wave.columns)
        print(dft1_pre.columns)
        print(dft1_iseg.columns)

        print("Found {} ORGretina4MModel events".format(len(dft1_gret)))
        print("Found {} ORGretina4MWaveformDecoder events".format(len(dft1_wave)))
        print("Found {} ORMJDPreAmpDecoderForAdc events".format(len(dft1_pre)))
        print("Found {} ORiSegHVCardDecoderForHV events".format(len(dft1_iseg)))


def make_processor_list():
    """ Make a list of processors to do to the data for the "tier one"
    (ie, gatified)
    """
    procs = pygama.TierOneProcessorList()

    procs.AddFromTier0("channel")
    procs.AddFromTier0("energy", output_name="onboard_energy")

    # Does order matter here? YES.
    # calculators (single-valued): pygama.processing.calculators
    # transforms (waveform-valued): pygama.processing.transforms

    procs.AddCalculator(pygama.processing.calculators.fit_baseline,
                        {"end_index":700},
                        output_name=["bl_slope", "bl_int"])

    procs.AddTransform(pygama.processing.transforms.remove_baseline,
                       {"bl_0":"bl_int", "bl_1":"bl_slope"},
                       output_waveform="blrm_wf")

    procs.AddTransform(pygama.processing.transforms.pz_correct,
                       {"rc":72},
                       input_waveform="blrm_wf",
                       output_waveform="pz_wf")

    procs.AddTransform(pygama.processing.transforms.trap_filter,
                       {"rampTime":200, "flatTime":400},
                       input_waveform="pz_wf",
                       output_waveform="trap_wf")

    procs.AddCalculator(pygama.processing.calculators.trap_max,
                        {},
                        input_waveform="trap_wf",
                        output_name="trap_max")

    procs.AddCalculator(pygama.processing.calculators.trap_max,
                        {"method":"fixed_time","pickoff_sample":400},
                        input_waveform="trap_wf",
                        output_name="trap_ft")

    return procs


def compareEnergy():
    from ROOT import TFile, TTree

    run = 72

    gFile = "./data/mjd_run%d.root" % run
    tf = TFile(gFile)
    tt = tf.Get("mjdTree")
    n = tt.Draw("trapE","","goff")
    trapE = tt.GetV1()
    trapE = np.asarray([trapE[i] for i in range(n)])
    x1, h1 = wl.GetHisto(trapE, 0, 10000, 10)

    print("ROOT file has %d hits", len(trapE))

    df = pd.read_hdf("%s/t2_run%d.h5" % (dataDir, run))

    x2, h2 = wl.GetHisto(df['onboard_energy'], 0, 3500000, 10000)
    x3, h3 = wl.GetHisto(df['trap_max'], 0, 1000000, 1000)
    x4, h4 = wl.GetHisto(df['trap_ft'], 0, 100, 1)

    print("pygama file has %d hits", len(df['onboard_energy']))

    # print(df.columns) 'channel', 'onboard_energy', 'bl_slope', 'bl_int', 'trap_max', 'trap_ft'

    energyK40 = 1460.820
    onboardE_K40 = 1.37958e6
    x2Cal = x2 * (energyK40 / onboardE_K40)

    trapE_K40 = 3626.64
    x1Cal = x1 * (energyK40 / trapE_K40)

    # plt.semilogy(x1, h1, ls='steps', c='b', lw=2, label="majorcaroot trapE")
    plt.semilogy(x1Cal, h1, ls='steps', c='b', lw=2, label="majorcaroot trapE (rough cal)")
    # plt.semilogy(x2, h2, ls='steps', c='r', lw=2, label="pygama onboard")
    plt.semilogy(x2Cal, h2, ls='steps', c='r', lw=2, label="pygama onboard (rough cal)")
    # plt.semilogy(x3, h3, ls='steps', c='g', lw=2, label="pygama trap_max")
    # plt.semilogy(x4, h4, ls='steps', c='m', lw=2, label="pygama trap_ft")

    plt.xlabel("Energy (uncalib.)",ha='right',x=1)
    plt.legend(loc=1)
    plt.show()


    # compare waveforms?
    bFile = "./data/OR_run%d.root" % run


if __name__=="__main__":
    main()