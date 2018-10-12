""" ========= PYGAMA =========
TIER 1 MAIN PROCESSING ROUTINE
Cythonized for maximum speed.
"""
cimport numpy as np

import os, re, sys
import numpy as np
import pandas as pd
import h5py

# import ipdb; ipdb.set_trace() # launch ipython debugger

from ..utils import update_progress
from ..decoders.digitizers import *


def ProcessTier1(filename,
                 processorList,
                 digitizer_list=None,
                 output_file_string="t2",
                 verbose=False,
                 output_dir=None):
    """
    Reads in "raw," or "tier 0," Orca data and saves to a hdf5 format using pandas
    filename: path to a tier1 data file
    processorList:
        TierOneProcessorList object with list of calculations/transforms you want done
        -- NOTE -- Order matters in the list! (Some calculations depend on others.)
    output_file_string: file is saved as <output_file_string>_run<runNumber>.h5
    verbose: spits out a progressbar to let you know how the processing is going
    """
    print("Starting pygama Tier 1 processing ...")
    print("  Input file: "+filename)

    directory = os.path.dirname(filename)
    output_dir = os.getcwd() if output_dir is None else output_dir

    # snag the run number (assuming filename ends in _run<number>.<filetype>)
    run_str = re.findall('run\d+', filename)[-1]
    runNumber = int(''.join(filter(str.isdigit, run_str)))

    # get pygama's available digitizers
    if digitizer_list is None:
        digitizer_list = get_digitizers()

    # get digitizers in the file
    f = h5py.File(filename, 'r')
    digitizer_list = [d for d in digitizer_list if d.decoder_name in f.keys()]

    print("   Found digitizers:")
    for d in digitizer_list:
        print("   -- {}".format(d.decoder_name))

    for digitizer in digitizer_list:
        print("Processing data from digitizer {}".format(digitizer.decoder_name))

        object_info = pd.read_hdf(filename, key=digitizer.class_name)
        digitizer.load_object_info(object_info)

        event_df = pd.read_hdf(filename, key=digitizer.decoder_name)
        # event_df = event_df[:1000]

        processorList.digitizer = digitizer
        processorList.max_event_number = len(event_df)
        processorList.verbose = verbose
        processorList.t0_cols = event_df.columns.values.tolist()
        processorList.runNumber = runNumber
        processorList.num = 0

        t1_df = event_df.apply(processorList.Process, axis=1)
        event_df = event_df.join(t1_df, how="left")
        processorList.DropTier0(event_df)

    if verbose:
        update_progress(1)

    # if verbose: print("Creating dataframe for file {}...".format(filename))
    # df_data = pd.DataFrame(appended_data)
    # df_data.set_index("event_number", inplace=True)

    t2_file_name = output_file_string + '_run{}.h5'.format(runNumber)
    t2_path = os.path.join(output_dir, t2_file_name)

    if verbose:
        print("Writing Tier 2 File:\n    {}".format(t2_path))

    event_df.to_hdf(
        t2_path,
        key="data",
        format='table',
        mode='w',
        data_columns=event_df.columns.tolist())

    print("Done.")

    return event_df