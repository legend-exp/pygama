import numpy as np
import pandas as pd
from abc import ABC

from .calculators import *
from .transforms import *
from ..utils import update_progress


class Tier1Processor(ABC):
    """
    Handle calculators and transforms.  Most are vectorized for super speed.
    Keep an internal 'intercom' of calculator results and waveform transforms.
    """
    def __init__(self, default_list=False):

        self.proc_list = []
        self.digitizer = None
        self.calcs = None # df for calcsulation results (no wfs)
        self.waves = {} # wfs only, NxM arrays (unpacked)

        if default_list:
            self.SetDefaultList()

    def Process(self, data_df, wfnames_out=None):
        """ Apply each processor to the Tier 0 input dataframe,
        and return a Tier 1 dataframe (i.e. gatified single-valued).
        Optionally return a dataframe with the waveform objects.
        """
        # save an ndarray of the waveforms, which are packed into cells
        self.waves["waveform"] = np.vstack([wf for wf in data_df['waveform']])

        # save the non-wf parts of the input dataframe separately
        data_cols = [col for col in data_df.columns if col != 'waveform']
        self.calcs = data_df[data_cols]

        # apply each processsor
        for processor in self.proc_list:
            print("Applying:", processor.function.__name__)

            p_result = processor.process(self.waves, self.calcs)

            if isinstance(processor, VectorCalculator):
                # calcs is updated inside the functions
                pass

            elif isinstance(processor, VectorTransformer):
                for wftype in p_result:
                    self.waves[wftype] = p_result[wftype]

        if wfnames_out is not None:
            wf_out = {wf : self.waves[wf] for wf in wfnames_out}
            return self.calcs, wf_out

        return self.calcs

    def AddCalculator(self, *args, **kwargs):
        self.proc_list.append(VectorCalculator(*args, **kwargs))

    def AddTransformer(self, *args, **kwargs):
        self.proc_list.append(VectorTransformer(*args, **kwargs))

    def SetDefaultList(self):
        self.AddCalculator(avg_baseline, fun_args = {"i_end":500})
        self.AddCalculator(fit_baseline, fun_args = {"i_end":500})
        self.AddTransformer(bl_subtract, fun_args = {"test":False})
        self.AddTransformer(trap_filter, fun_args = {"test":False})


class ProcessorBase(ABC):
    """ base class for Tier 1 processors.
    - calculators.py - calculate single values from a waveform
    - transforms.py - create a new waveform from a waveform
    """
    def __init__(self, function, fun_args={}):
        """ save some argunemnts specific to this transform/calculator """
        self.function = function
        self.fun_args = fun_args # so fun

    def process(self, waves, calcs):
        """ run the given calcsulation on the wf block in waves.
        can also use results from other calculations via calcs.
        individual processor functions can decide if they want to use the
        df axes, or convert to a numpy array for extra speed
        """
        return self.function(waves, calcs, **self.fun_args)


class VectorCalculator(ProcessorBase):
    def __init__(self, function, fun_args={}):
        super().__init__(function, fun_args)


class VectorTransformer(ProcessorBase):
    def __init__(self, function, fun_args={}):
        super().__init__(function, fun_args)

