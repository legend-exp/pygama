import numpy as np
import pandas as pd
from pprint import pprint
import pygama.dsp.calculators as pc
import pygama.dsp.transforms as pt
from ..utils import update_progress

class Processor:
    """
    base class for Tier 1 processors.
    - calculators.py - calculate single values from a waveform
    - transforms.py - create a new waveform from a waveform
    """
    def __init__(self, function, fun_args={}):
        """
        save some arguments specific to this transform/calculator
        """
        self.function = function
        self.fun_args = fun_args # so fun

    def process_block(self, waves, calcs):
        """
        run the given calculation. we always pass in:
        `waves` : a dict of "waveform blocks", i.e. 2d numpy arrays
        `calcs` : a pd.DataFrame with single-valued calculator results
        """
        return self.function(waves, calcs, **self.fun_args)


class Calculator(Processor):
    def __init__(self, function, fun_args={}):
        super().__init__(function, fun_args)
        # may want to declare Calculator-specific stuff here at some point


class Transformer(Processor):
    def __init__(self, function, fun_args={}):
        super().__init__(function, fun_args)
        # may want to declare Transformer-specific stuff here at some point


class Intercom:
    """
    we input a list of calculators and transforms, and
    this class manages an "intercom" consisting of:
    `waves` : a dict of 2d np.array's holding the raw wfs and their Transforms
    `calcs` : a pd.DataFrame with single-valued calculator results,

    the processors are run on each wf block in the order you defined them,
    so if you want one to depend on the result of another,
    order your input list accordingly.
    """
    def __init__(self, settings=None, default_list=False):

        self.proc_list = []
        self.calcs = None # df for calculation results (no wfs)
        self.waves = {} # wfs only, NxM arrays (unpacked)
        self.digitizer = None # may need for card-specifics like nonlinearity

        # parse the JSON settings to create a list of processors and options
        if settings is not None:
            self.settings = settings
            for key in settings:

                # handle 2nd pass processors
                if "pass2" in key:
                    name = "".join(key.split("_")[:-1])
                else:
                    name = key

                if isinstance(settings[key], dict):
                    self.add(name, settings[key])

                # handle multiple instances of a calculator w/ diff params
                elif isinstance(settings[key], list):
                    for i, d2 in enumerate(settings[key]):
                        self.add("{}-{}".format(name, i), d2)

        elif default_list:
            self.set_default_list()
        else:
            print("Warning: no processors set!")

        # trick to pass in settings to the Processors w/o an extra argument
        self.waves["settings"] = self.settings


    def add(self, fun_name, settings={}):
        """
        add a new processor to the list,
        with a string name and a dict of settings
        """
        fun_name = fun_name.split("-")[0] # handle multiple instances
        # print("adding", fun_name, settings)

        if fun_name in dir(pc):
            self.proc_list.append(Calculator(getattr(pc, fun_name), settings))
        elif fun_name in dir(pt):
            self.proc_list.append(Transformer(getattr(pt, fun_name), settings))
        else:
            print("ERROR! unknown function:", fun_name)
            exit()


    def set_default_list(self):
        """
        use a minimal sequence of processors
        """
        for proc in ["fit_bl", "bl_sub", "trap", "get_max"]:
            settings = self.settings[proc] if proc in self.settings else {}
            self.add(proc, settings)


    def set_intercom(self, data_df):
        """
        declare self.waves and self.calcs, our intercom data objects.
        """
        cols = data_df.columns.values
        wf_start = np.where(cols == 0)[0][0]
        wf_stop = len(cols)-1
        self.waves["waveform"] = data_df.iloc[:, wf_start:wf_stop].values
        self.calcs = data_df.iloc[:, 0: wf_start-1].copy()

        if "blest" in self.waves["settings"]["blsub"]:
          if self.waves["settings"]["blsub"]["blest"] == "fcdaq":
            self.calcs["fcdaq"] = data_df.bl.values

    def process(self, data_df, verbose=False, wfnames_out=None):
        """
        Apply each processor to the Tier 1 input dataframe,
        and return a Tier 2 dataframe (i.e. gatified single-valued).
        Optionally return a dataframe with the waveform objects.
        """
        self.set_intercom(data_df)

        for processor in self.proc_list:

            if verbose:
                print(" -> ", processor.function.__name__, processor.fun_args)

            p_result = processor.process_block(self.waves, self.calcs)

            if isinstance(processor, Calculator):
                # self.calcs is updated inside the functions right now
                pass

            elif isinstance(processor, Transformer):
                for wftype in p_result:
                    self.waves[wftype] = p_result[wftype]

        if wfnames_out is not None:
            wf_out = {wf : self.waves[wf] for wf in wfnames_out}
            return self.calcs, wf_out

        return self.calcs


