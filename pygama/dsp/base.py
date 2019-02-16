import numpy as np
import pandas as pd
from abc import ABC
import pygama.dsp.calculators as pc
import pygama.dsp.transforms as pt
from ..utils import update_progress


class Tier1Processor(ABC):
    """
    Handle a list of Tier 1 calculators and transforms.
    Keep an internal 'intercom' of calculator results and waveform transforms.
    """
    def __init__(self, settings=None, default_list=False):

        self.proc_list = []
        self.calcs = None # df for calculation results (no wfs)
        self.waves = {} # wfs only, NxM arrays (unpacked)
        self.digitizer = None # may need for card-specifics like nonlinearity

        # add a list of processors and options
        if settings is not None:
            self.settings = settings
            for key in settings:
                self.add(key, settings[key])

        if default_list:
            self.set_default_list()


    def set_intercom(self, data_df):
        """
        declare self.waves and self.calcs, our intercom data objects.
        save waveforms as an ndarray into the dict self.waves.
        then save the single-values parts of the input dataframe separately
        into self.calcs, which we build on to create a Tier 2 dataframe
        (i.e. gatified single-valued.)
        """
        cols = data_df.columns.values
        wf_start = np.where(cols == 0)[0][0]
        wf_stop = len(cols)-1
        self.waves["waveform"] = data_df.iloc[:, wf_start:wf_stop].values
        self.calcs = data_df.iloc[:, 0: wf_start-1].copy()


    def process(self, data_df, verbose=False, wfnames_out=None):
        """
        Apply each processor to the Tier 1 input dataframe,
        and return a Tier 2 dataframe (i.e. gatified single-valued).
        Optionally return a dataframe with the waveform objects.
        """
        self.set_intercom(data_df)

        for processor in self.proc_list:

            if verbose:
                print("Applying:", processor.function.__name__)

            p_result = processor.process_block(self.waves, self.calcs)

            if isinstance(processor, Calculator):
                # self.calcs is updated inside the functions
                pass

            elif isinstance(processor, Transformer):
                for wftype in p_result:
                    self.waves[wftype] = p_result[wftype]

        if wfnames_out is not None:
            wf_out = {wf : self.waves[wf] for wf in wfnames_out}
            return self.calcs, wf_out

        return self.calcs


    def add(self, fun_name, settings={}):
        """
        add a new processor to the list, with a string name and a
        dict of settings, which overrides any other settings we've already set
        """
        # get the settings
        if fun_name in self.settings:
            self.settings[fun_name] = {**self.settings[fun_name], **settings}
        else:
            self.settings[fun_name] = settings

        # add the processor
        if fun_name in dir(pc):
            self.proc_list.append(Calculator(getattr(pc, fun_name), self.settings[fun_name]))

        elif fun_name in dir(pt):
            self.proc_list.append(Transformer(getattr(pt, fun_name), self.settings[fun_name]))
        else:
            print("ERROR! unknown function:", fun_name)
            sys.exit()


    def set_default_list(self):
        for proc in ["fit_baseline", "bl_subtract", "trap_filter", "trap_max"]:
            settings = self.settings[proc] if proc in self.settings else {}
            self.add(proc, settings)


class ProcessorBase(ABC):
    """ base class for Tier 1 processors.
    - calculators.py - calculate single values from a waveform
    - transforms.py - create a new waveform from a waveform
    """
    def __init__(self, function, fun_args={}):
        """ save some argunemnts specific to this transform/calculator """
        self.function = function
        self.fun_args = fun_args # so fun

    def process_block(self, waves, calcs):
        """ run the given calculation on the wf block in waves.
        can also use results from other calculations via calcs.
        individual processor functions can decide if they want to use the
        df axes, or convert to a numpy array for extra speed
        """
        return self.function(waves, calcs, **self.fun_args)


class Calculator(ProcessorBase):
    def __init__(self, function, fun_args={}):
        super().__init__(function, fun_args)
        # may want to declare Calculator-specific stuff here at some point


class Transformer(ProcessorBase):
    def __init__(self, function, fun_args={}):
        super().__init__(function, fun_args)
        # may want to declare Transformer-specific stuff here at some point
