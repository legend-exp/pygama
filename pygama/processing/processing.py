""" PUBLIC PROCESSING METHODS FOR TIERS 0 and 1"""
import os, glob
import numpy as np
from ._tier0 import ProcessTier0
from ._tier1 import ProcessTier1
from .base_classes import TierOneProcessorList

__all__ = ["get_default_processor_list", "process_tier_0", "process_tier_1"]


def process_tier_0(datadir,
                   runList,
                   verbose=True,
                   output_dir=None,
                   chan_list=None,
                   n_max=np.inf):
    """ Wrapper function for ProcessTier0 """

    for run in runList:

        filenameList = glob.glob(os.path.join(datadir, "*Run{}".format(run)))
        if len(filenameList) == 0:
            print("No file with name Run{} in directory {}! Skipping run...".
                  format(run, datadir))
            continue
        elif len(filenameList) > 1:
            print(
                "More than one file with name Run{} in directory {}! Skipping run..."
                .format(run, datadir))
            continue
        filename = filenameList[0]
        filepath = os.path.join(datadir, filename)

        ProcessTier0(
            filepath,
            verbose=verbose,
            output_dir=output_dir,
            n_max=n_max,
            chan_list=chan_list)


def process_tier_1(datadir,
                   runList,
                   processor_list,
                   verbose=True,
                   output_dir=None,
                   output_file_string="t2",
                   num_threads=1,
                   overwrite=True):
    """ Wrapper function for ProcessTier1 """

    # if processor_list is None:
    #     processor_list = get_default_processor_list()

    t1_args = []
    for run in runList:
        filepath = os.path.join(datadir, "t1_run{}.h5".format(run))
        if not overwrite:
            outfilepath = os.path.join(
                output_dir, output_file_string + "_run{}.h5".format(run))
            if os.path.isfile(outfilepath):
                print("Skipping run {} because t2 file already created...".
                      format(run))
                continue

        if num_threads == 1:
            ProcessTier1(
                filepath,
                processor_list,
                verbose=verbose,
                output_dir=output_dir,
                output_file_string=output_file_string)
        else:
            t1_args.append([filepath, processor_list])
            keywords = {"verbose": verbose, "output_dir": output_dir}

    if num_threads > 1:
        max_proc = cpu_count()  # careful, its a lot to load in RAM...
        num_threads = num_threads if num_threads < max_proc else max_proc
        p = Pool(num_threads)
        # p.starmap( partial(ProcessTier0, **keywords), t0_args)
        p.starmap(partial(ProcessTier1, **keywords), t1_args)


def get_default_processor_list():
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


