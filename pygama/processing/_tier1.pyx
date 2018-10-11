""" ========= PYGAMA =========
TIER 1 MAIN PROCESSING ROUTINE
Cythonized for maximum speed.
"""
cimport numpy as np

import numpy as np
import os, re
import pandas as pd
import h5py

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
    processorList: TierOneProcessorList object with list of calculations/transforms you want done
    output_file_string: file is saved as <output_file_string>_run<runNumber>.h5
    verbose: spits out a progressbar to let you know how the processing is going
    """
    print("Starting pygama Tier 1 processing ...")

    directory = os.path.dirname(filename)
    output_dir = os.getcwd() if output_dir is None else output_dir

    #snag the run number (assuming filename ends in _run<number>.<filetype>)
    run_str = re.findall('run\d+', filename)[-1]
    runNumber = int(''.join(filter(str.isdigit, run_str)))

    if digitizer_list is None:
        #digitize everything available
        digitizer_list = get_digitizers()
    digitizer_decoder_names = [d.class_name for d in digitizer_list]

    #find the available keys
    f = h5py.File(filename, 'r')
    for d in digitizer_list:
        if d.decoder_name not in f.keys():
            digitizer_list.remove(d)

    print("\nBeginning Tier 1 processing of file {}...".format(filename))

    for digitizer in digitizer_list:
        print("   Processing from digitizer {}".format(digitizer.class_name))

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

    if verbose: update_progress(1)

    # if verbose: print("Creating dataframe for file {}...".format(filename))
    # df_data = pd.DataFrame(appended_data)
    # df_data.set_index("event_number", inplace=True)

    t2_file_name = output_file_string + '_run{}.h5'.format(runNumber)
    t2_path = os.path.join(output_dir, t2_file_name)

    if verbose:
        print("\nWriting {} to tier2 file {}...".format(filename, t2_path))

    event_df.to_hdf(
        t2_path,
        key="data",
        format='table',
        mode='w',
        data_columns=event_df.columns.tolist())
    return event_df
