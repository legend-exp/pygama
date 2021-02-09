#!/usr/bin/env python3
import os, glob, time
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
plt.style.use('./pltReports.mplstyle')

import pygama
import waveLibs as wl

# import ipdb; ipdb.set_trace() # launch ipython debugger

# data_dir = "/Users/wisecg/project/mj60/data"
data_dir = "/Users/wisecg/dev/mj60/data"

def main():

    # process_MJD()
    process_MJ60()
    # compareEnergy_MJ60()
    # compareEnergy_MJD()
    # time_processing_MJD()
    # check_profile()


def process_t0(run, data_dir, n_max=np.inf, verbose=False):
    """ Tier 0 defined in:
    - pygama/processing/_daq_to_raw.pyx
    """
    run_list = [run]
    pygama.process_tier_0(data_dir, run_list, output_dir=data_dir, chan_list=None,
                          n_max=n_max, verbose=True)


def process_t1(run, data_dir, n_cpu=1):
    """ Tier 1 defined in:
    - pygama/processing/_raw_to_dsp.pyx
    Requires a processor list.
    if run_list is > 1 run, can try processing each run on a separate thread
    with num_threads > 1.  Careful, it's a lot to load in RAM ...
    """
    run_list = [run]
    pygama.process_tier_1(data_dir, run_list, make_processor_list(),
                          output_dir=data_dir, num_threads=n_cpu)


def sh(cmd, sh=False):
    """ Wraps a shell command."""
    import shlex
    import subprocess as sp
    if not sh: sp.call(shlex.split(cmd))  # "safe"
    else:      sp.call(cmd, shell=sh)     # "less safe"


def process_MJD():
    """ process a random MJD background run """

    # run = 36854 # random bg
    run = 42343 # random cal
    n_max = 5000

    # regular processing

    # process data with majorcaroot / GAT
    # raw_file = glob.glob("./data/*Run{}".format(run))[0].split("/")[-1]
    # pwd = os.getcwd()
    # os.chdir("./data")
    # sh("majorcaroot {}".format(raw_file))
    # sh("process_mjd_data_p1 OR_run{}.root".format(run))
    # os.chdir(pwd)
    # return

    # process data with pygama
    # process_t0(run, data_dir)#, n_max)
    # print_daq_to_raw_info(run)
    # process_t1(run, data_dir, n_cpu=2)
    # return

    # make an energy plot
    dft2 = pd.read_hdf("%s/t2_run%d.h5" % (data_dir, run))
    plt.hist(dft2['trap_max'], bins=300, histtype="step", log=True)
    plt.xlabel("trap_max", ha='right', x=1)
    plt.ylabel("counts", ha='right', y=1)
    plt.savefig("./plots/mjd_energy.pdf")

    # this class reads the "ORGretina4MWaveformDecoder" dataframe
    ft1 = "%s/t1_run%d.h5" % (data_dir, run)
    dc = pygama.Gretina4M(ft1)
    df_events = pd.read_hdf(ft1, key=dc.decoder_name)

    # plot a few waveforms
    plt.figure(figsize=(8,5))
    plt.cla()
    nwf = 100
    for i, (index, row) in enumerate(df_events.iterrows()):
        wf = dc.parse_event_data(row)
        # plt.plot(wf.data[0][4:]) # mj60
        plt.plot(wf.data)
        if i > nwf:
            break
    plt.xlabel("Time [ns]", ha='right', x=1)
    plt.ylabel("ADC", ha='right', y=1)
    # plt.show()
    plt.savefig("./plots/mjd_waveforms.png")


def process_MJ60():
    # process mj60 run (struck digitizer)

    run = 72 # bkg data, with buffer wrap
    # run = 74 # hv pulser, no buffer wrap mode

    # process data
    # process_t0(run, np.inf, data_dir, verbose=True)
    # print_daq_to_raw_info(run)
    process_t1(run, data_dir)
    return

    # make an energy plot
    dft2 = pd.read_hdf("%s/t2_run%d.h5" % (data_dir, run))
    plt.hist(dft2['trap_max'], bins=300, histtype="step", log=True)
    plt.xlabel("trap_max", ha='right', x=1)
    plt.ylabel("counts", ha='right', y=1)
    plt.savefig("./plots/mj60_energy_run%d.pdf" % run)

    # raw_to_dsp file
    ft1 = "%s/t1_run%d.h5" % (data_dir, run)

    print("available Tier 1 keys:")
    store = pd.HDFStore(ft1)
    for g in store.groups():
        print("  ", g)
    store.close()

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


def print_daq_to_raw_info(run):
    """ The pygama Tier 1 HDF5 contains multiple datasets,
    whose keys must be specified when we do pd.read_hdf(.., key="name")
    Tier 2 doesn't have this structure, so the key is not needed.
    """
    print("\n\nDone with processing .... Printing tier info .... run",run)

    # get a list of keys
    print("Available Tier 1 keys:")
    store = pd.HDFStore("%s/t1_run%d.h5" % (data_dir, run))
    for g in store.groups():
        print("  ", g)
    store.close()

    # MJ60
    if run == 74:

        dft1_sis = pd.read_hdf("{}/t1_run{}.h5".format(data_dir, run), key="ORSIS3302DecoderForEnergy")
        dft1_mod = pd.read_hdf("{}/t1_run{}.h5".format(data_dir, run), key="ORSIS3302Model")

        print(dft1_sis.columns)
        print(dft1_mod.columns)

        print("Found {} ORSIS3302DecoderForEnergy events".format(len(dft1_sis)))
        print("Found {} ORSIS3302Model events".format(len(dft1_mod)))

    # MJD
    if run == 36854:

        dft1_gret = pd.read_hdf("%s/t1_run%d.h5" % (data_dir, run), key="ORGretina4MModel")
        dft1_wave = pd.read_hdf("%s/t1_run%d.h5" % (data_dir, run), key="ORGretina4MWaveformDecoder")
        dft1_pre = pd.read_hdf("%s/t1_run%d.h5" % (data_dir, run), key="ORMJDPreAmpDecoderForAdc")
        dft1_iseg = pd.read_hdf("%s/t1_run%d.h5" % (data_dir, run), key="ORiSegHVCardDecoderForHV")

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

    # procs.AddFromTier0("channel")
    # procs.AddFromTier0("energy", output_name="onboard_energy")

    # Does order matter here? YES.
    # calculators (single-valued): pygama.processing.calculators
    # transforms (waveform-valued): pygama.processing.transforms

    # procs.AddCalculator(pygama.processing.calculators.fit_baseline,
    #                     {"end_index":700},
    #                     output_name=["bl_slope", "bl_int"])
    #
    # procs.AddTransform(pygama.processing.transforms.remove_baseline,
    #                    {"bl_0":"bl_int", "bl_1":"bl_slope"},
    #                    output_waveform="blrm_wf")
    #
    # procs.AddTransform(pygama.processing.transforms.pz_correct,
    #                    {"rc":72},
    #                    input_waveform="blrm_wf",
    #                    output_waveform="pz_wf")
    #
    # procs.AddTransform(pygama.processing.transforms.trap_filter,
    #                    {"rampTime":200, "flatTime":400},
    #                    input_waveform="pz_wf",
    #                    output_waveform="trap_wf")
    #
    # procs.AddCalculator(pygama.processing.calculators.trap_max,
    #                     {},
    #                     input_waveform="trap_wf",
    #                     output_name="trap_max")
    #
    # procs.AddCalculator(pygama.processing.calculators.trap_max,
    #                     {"method":"fixed_time","pickoff_sample":400},
    #                     input_waveform="trap_wf",
    #                     output_name="trap_ft")

    return procs


def default_processor_list():
    """" Make a list of processors to do to the data for the "tier one"
    (ie, gatified)"""

    procs = TierOneProcessorList()

    #pass energy thru to t1
    # procs.AddFromTier0("energy")
    procs.AddFromTier0("channel")
    procs.AddFromTier0("energy", "onboard_energy")

    #is the wf saturated?
    procs.AddCalculator(is_saturated, {}, output_name="is_saturated")

    #baseline remove
    procs.AddCalculator(
        fit_baseline, {"end_index": 700}, output_name=["bl_slope", "bl_int"])
    procs.AddTransform(
        remove_baseline, {
            "bl_0": "bl_int",
            "bl_1": "bl_slope"
        },
        output_waveform="blrm_wf")

    #calculate max currents from baseline-removed wf with a few different sigma vals
    for sig in [1, 3, 5, 7]:
        procs.AddCalculator(
            current_max, {"sigma": sig},
            input_waveform="blrm_wf",
            output_name="current_max_{}".format(sig))

    #calculate a few time points (50%, 90%, 95%)
    for tp in [0.5, 0.9, 0.95]:
        procs.AddCalculator(
            calc_timepoint, {"percentage": tp},
            input_waveform="blrm_wf",
            output_name="tp_{:.0f}".format(tp * 100))

    #estimate t0
    procs.AddTransform(
        savgol_filter, {
            "window_length": 47,
            "order": 2
        },
        input_waveform="blrm_wf",
        output_waveform="sg_wf")
    procs.AddCalculator(
        t0_estimate, {}, input_waveform="sg_wf", output_name="t0est")

    #energy estimator: pz correct, calc trap
    procs.AddTransform(
        pz_correct, {"rc": 72}, input_waveform="sg_wf", output_waveform="pz_wf")
    procs.AddTransform(
        trap_filter, {
            "rampTime": 200,
            "flatTime": 400
        },
        input_waveform="pz_wf",
        output_waveform="trap_wf")

    procs.AddCalculator(
        trap_max, {}, input_waveform="trap_wf", output_name="trap_max")
    procs.AddCalculator(
        trap_max, {
            "method": "fixed_time",
            "pickoff_sample": 400
        },
        input_waveform="trap_wf",
        output_name="trap_ft")

    procs.AddCalculator(
        fit_baseline, {
            "start_index": 1150,
            "end_index": -1,
            "order": 0
        },
        input_waveform="pz_wf",
        output_name="ft_mean")
    procs.AddCalculator(
        fit_baseline, {
            "start_index": 1150,
            "end_index": -1,
            "order": 1
        },
        input_waveform="pz_wf",
        output_name=["ft_slope", "ft_int"])

    return procs


def compareEnergy_MJ60():
    from ROOT import TFile, TTree

    run = 72

    gFile = "./data/mjd_run%d.root" % run
    tf = TFile(gFile)
    tt = tf.Get("mjdTree")
    n = tt.Draw("trapE","","goff")
    trapE = tt.GetV1()
    trapE = np.asarray([trapE[i] for i in range(n)])
    x1, h1 = wl.GetHisto(trapE, 0, 10000, 10)

    print("ROOT file has {} hits".format(len(trapE)))

    df = pd.read_hdf("%s/t2_run%d.h5" % (data_dir, run))

    x2, h2 = wl.GetHisto(df['onboard_energy'], 0, 3500000, 4000)
    x3, h3 = wl.GetHisto(df['trap_max'], 0, 1000000, 1000)
    x4, h4 = wl.GetHisto(df['trap_ft'], 0, 100, 1)

    print("pygama file has {} hits".format(len(df['onboard_energy'])))

    # print(df.columns) 'channel', 'onboard_energy', 'bl_slope', 'bl_int', 'trap_max', 'trap_ft'

    energyK40 = 1460.820
    onboardE_K40 = 1.37958e6
    x2Cal = x2 * (energyK40 / onboardE_K40)

    trapE_K40 = 3626.64
    x1Cal = x1 * (energyK40 / trapE_K40)

    print("majorcaroot: {:.2f} kev/bin  pygama: {:.2f} kev/bin".format(x1Cal[1]-x1Cal[0], x2Cal[1]-x2Cal[0]))

    plt.semilogy(x1Cal, h1, ls='steps', c='b', lw=2, label="majorcaroot trapE (rough cal)")
    plt.semilogy(x2Cal, h2, ls='steps', c='r', lw=2, label="pygama onboard (rough cal)")

    print("majorcaroot evt sum: {}  pygama evt sum: {}".format(np.sum(h1), np.sum(h2)))

    # plt.semilogy(x1, h1, ls='steps', c='b', lw=2, label="majorcaroot trapE")
    # plt.semilogy(x2, h2, ls='steps', c='r', lw=2, label="pygama onboard")
    # plt.semilogy(x3, h3, ls='steps', c='g', lw=2, label="pygama trap_max")
    # plt.semilogy(x4, h4, ls='steps', c='m', lw=2, label="pygama trap_ft")

    plt.xlabel("Energy (uncalib.)",ha='right',x=1)
    plt.legend(loc=1)
    plt.show()


def compareEnergy_MJD():
    from ROOT import TFile, TTree

    run = 36854

    gFile = "./data/mjd_run%d.root" % run
    tf = TFile(gFile)
    tt = tf.Get("mjdTree")
    n = tt.Draw("trapE","","goff")
    trapE = tt.GetV1()
    trapE = np.asarray([trapE[i] for i in range(n)])
    x1, h1 = wl.GetHisto(trapE, 0, 10000, 10)

    print("ROOT file has {} hits".format(len(trapE)))

    df = pd.read_hdf("%s/t2_run%d.h5" % (data_dir, run))
    print("pygama file has {} hits".format(len(df)))

    x2, h2 = wl.GetHisto(df['onboard_energy'], 0, 10000, 10)
    x3, h3 = wl.GetHisto(df['trap_max'], 0, 10000, 10)
    x4, h4 = wl.GetHisto(df['trap_ft'], 0, 100, 1)

    plt.semilogy(x1, h1, ls='steps', c='b', lw=2, label="majorcaroot trapE")
    plt.semilogy(x2, h2, ls='steps', c='r', lw=2, label="pygama onboard")
    plt.semilogy(x3, h3, ls='steps', c='g', lw=2, label="pygama trap_max")
    # plt.semilogy(x4, h4, ls='steps', c='m', lw=2, label="pygama trap_ft")

    plt.xlabel("Energy (uncalib.)",ha='right',x=1)
    plt.legend(loc=1)
    plt.show()


def time_processing_MJD():
    # timed processing

    run = 42343

    # raw_file = glob.glob("./data/*Run{}".format(run))[0].split("/")[-1]
    # pwd = os.getcwd()
    # os.chdir("./data")
    # t_build = time.time()
    # sh("majorcaroot {}".format(raw_file))
    # print("Time to build: {:.2f}".format(time.time() - t_build))
    # t_gat = time.time()
    # sh("process_mjd_data_p1 OR_run{}.root".format(run))
    # print("Time to simple-gatify: {:.2f}".format(time.time() - t_gat))
    # print("Total Elapsed: {:.2f} sec".format(time.time() - t_build))
    # os.chdir(pwd)
    # exit()

    # GAT results
    # Time to build: 59.33
    # Time to simple-gatify: 32.24
    # Total Elapsed: 91.57 sec

    # process data with pygama
    # t1 = time.time()
    # process_t0(run, data_dir)#, n_max)
    # print("Tier 0 elapsed: {:.2f}".format(time.time() - t1))
    t2 = time.time()
    process_t1(run, data_dir, n_cpu=2)
    print("Tier 1 elapsed: {:.2f}".format(time.time() - t2))
    # print("Total time: {:.2f}".format(time.time() - t1))

    # pygama initial results
    # Tier 1: Time elapsed: 81.45 sec
    # Tier 2: Time elapsed: 746.71 sec  (0.00153 sec/wf)


def check_profile():

    from pathlib import Path
    pyfiles = list(Path("/Users/wisecg/dev/pygama/").rglob("*.py"))
    pyfiles.extend(list(Path("/Users/wisecg/dev/pygama/").rglob("*.pyx")))
    pyfiles = [str(f).split("/")[-1] for f in pyfiles]
    print(pyfiles)

    with open("./profile_results.txt") as f:
        profile = f.readlines()

    for line in profile:
        if any(f in line for f in pyfiles):
            print(line.rstrip())



if __name__=="__main__":
    # main()
    import cProfile
    cProfile.run('main()', "test.profile")
