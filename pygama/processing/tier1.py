""" pygama tier 1 processing
tier 1 data --> DSP --> tier 2 (i.e. gatified)
"""
import os, re, sys, time
import numpy as np
import pandas as pd

from .tier0 import get_decoders
from ..decoders.data_loading import *
from ..decoders.digitizers import *
from ..utils import *

def ProcessTier1(t1_file,
                 processor_list,
                 digitizer_list=None,
                 out_prefix="t2",
                 verbose=False,
                 out_dir=None,
                 multiprocess=True):
    """
    the processor_list is:
    - TierOneProcessor object with list of calculations/transforms you want done
    - Order matters in the list! (Some calculations depend on others.)
    """
    print("Starting pygama Tier 1 processing ...")
    print("   Input file: {}".format(t1_file))

    start = time.clock()
    in_dir = os.path.dirname(t1_file)
    out_dir = os.getcwd() if out_dir is None else out_dir

    # with pd.HDFStore(t1_file,'r') as store:
        # print(store.keys())

    if digitizer_list is None:
        digitizer_list = get_decoders()
        # digitizer_list = get_digitizers()
        print(digitizer_list)




    # withs

    sys.exit()




    if digitizer_list is None:
        digitizer_list = get_digitizers()

    f = h5py.File(t1_file, 'r')
    digitizer_list = [d for d in digitizer_list if d.decoder_name in f.keys()]

    print("   Found digitizers:")
    for d in digitizer_list:
        print("   -- {}".format(d.decoder_name))


    for digitizer in digitizer_list:
        print("Processing data from digitizer {}".format(digitizer.decoder_name))

        object_info = pd.read_hdf(t1_file, key=digitizer.class_name)

        digitizer.load_metadata(object_info)


def get_digitizers():
    return [sub() for sub in Digitizer.__subclasses__()]
