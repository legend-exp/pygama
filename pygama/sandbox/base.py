""" Base processor classes for pygama """

import os, glob
import numpy as np
import pandas as pd
from future.utils import iteritems
from abc import ABC, abstractmethod
from pygama.utils import update_progress


class ProcessorBase(ABC):
    """ classes that wrap functional implementations of
    calculators or transformers """

    def __init__(self,
                 function,
                 output_name,
                 perm_args={},
                 input_waveform="waveform"):

        self.function = function
        self.output_name = output_name
        self.perm_args = perm_args
        self.input_waveform_name = input_waveform

    #replaces arg by name with values from the existing df
    def replace_args(self, event_data):
        # check args list for string vals which match keys in param dict

        # copy we'll actually pass to the function
        self.args = self.perm_args.copy()

        # print(self.function.__name__, self.perm_args)

        for (arg, val) in iteritems(self.args):
            #   print(arg, val)
            #
            #   if getattr(val, '__iter__', False):
            #     self.args[arg] = val
            try:
                if val in event_data.keys():
                    self.args[arg] = event_data[val]
            except TypeError:  #For when we have an array -- like passing in an array of time points
                pass
        # print ("--> ", self.args)

    def set_waveform(self, waveform_dict):
        self.input_wf = waveform_dict[self.input_waveform_name]

    #This will work for a Calculator or Transformer, not a DB Lookup? (which I've broken for now)
    def process(self):
        return self.function(self.input_wf, **self.args)


class Calculator(ProcessorBase):

    def __init__(self, function, output_name, args={},
                 input_waveform="waveform"):
        super().__init__(function, output_name=output_name,
            perm_args=args, input_waveform=input_waveform)


class Transformer(ProcessorBase):

    def __init__(self, function, output_waveform, args={},
                 input_waveform="waveform"):
        super().__init__(
            function, output_name=output_waveform, perm_args=args,
            input_waveform=input_waveform)


# class DatabaseLookup(ProcessorBase):
#
#     def __init__(self, function, args={}, output_name=None):
#         print(
#             "Database Lookup has been murdered in cold blood.
#              Either get it working or remove the DB call from your processor.
#              B. Shanks, 8/15/18."
#         )
#         sys.exit()
#
#         self.function = function
#
#         self.output_name = output_name
#
#       def replace_args(self, param_dict):
#         #check args list for string vals which match keys in param dict
#         self.args = self.perm_args.copy() #copy we'll actually pass to the function
#
#         for (arg, val) in iteritems(self.args):
#           if val in param_dict.keys():
#             self.args[arg] = param_dict[val]
#
#       def process(self):
#         return self.function(**self.args)


class Tier0Passer(ProcessorBase):

    def __init__(self, t0_name, output_name=None):
        self.t0_name = t0_name
        if output_name is None: output_name = t0_name
        self.output_name = output_name

    def replace_args(self, param_dict):
        self.t0_value = param_dict[self.t0_name]

    def process(self):
        return self.t0_value


class TierOneProcessorList():
    """
    Class to handle the list of transforms/calculations we do in the processing
    """
    def __init__(self):
        self.list = []
        self.waveform_dict = {}
        self.event_data = None
        self.digitizer = None
        self.runNumber = 0

        # t1 fields to make available for t2 processors
        self.t0_map = {}

    def Process(self, t0_row, flat=False):
        if self.verbose and self.num % 100 == 0:
            if (float(self.num) / self.max_event_number) < 1:
                update_progress(
                    float(self.num) / self.max_event_number, self.runNumber)

        # clear things out from the last waveform
        if flat:
            waveform = t0_row
            self.waveform_dict = {"waveform": waveform}
        else:
            waveform = self.digitizer.parse_event_data(t0_row)
            self.waveform_dict = {"waveform": waveform.get_waveform()}

        new_cols = {}
        try:
            new_cols["fs_start"] = waveform.full_sample_range[0]
            new_cols["fs_end"] = waveform.full_sample_range[1]
        except AttributeError:
            #in case it isn't a multisampled waveform object
            pass

        # Apply each processor
        for processor in self.list:
            processor.replace_args(new_cols)
            processor.set_waveform(self.waveform_dict)
            # print("now i'm using " +processor.function.__name__)

            if isinstance(processor, Transformer):
                self.waveform_dict[processor.output_name] = processor.process()
            else:
                output = processor.output_name
                calc = processor.process()

                # Handles multiple outputs for, e.g., time point calculations
                if not isinstance(output, str) and len(output) > 1:
                    for i, out in enumerate(output):
                        new_cols[out] = calc[i]
                else:
                    new_cols[output] = calc
        self.num += 1

        new_cols_series = pd.Series(new_cols)
        return new_cols_series

    def AddTransform(self,
                     function,
                     args={},
                     input_waveform="waveform",
                     output_waveform=None):
        self.list.append(
            Transformer(
                function,
                args=args,
                input_waveform=input_waveform,
                output_waveform=output_waveform))

    def AddCalculator(self,
                      function,
                      args={},
                      input_waveform="waveform",
                      output_name=None):
        self.list.append(
            Calculator(
                function,
                args=args,
                input_waveform=input_waveform,
                output_name=output_name))

    # def AddDatabaseLookup(self, function, args={}, output_name=None):
    #   self.list.append( DatabaseLookup(function, args, output_name) )

    def AddFromTier0(self, name, output_name=None):
        if output_name is None:
            output_name = name
        self.t0_map[name] = output_name
        # self.list.append( Tier0Passer(name, output_name) )

    def DropTier0(self, df_t1):
        df_t1.rename(columns=self.t0_map, inplace=True)
        drop_cols = []
        for t0_col in self.t0_cols:
            if t0_col not in self.t0_map.keys():
                drop_cols.append(t0_col)
        df_t1.drop(drop_cols, axis=1, inplace=True)


