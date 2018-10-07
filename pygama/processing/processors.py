from future.utils import iteritems
from abc import ABC, abstractmethod

class ProcessorBase(ABC):
    def __init__(self, function, output_name, perm_args = {}, input_waveform="waveform"):
        self.function = function
        self.output_name = output_name
        self.perm_args = perm_args
        self.input_waveform_name = input_waveform

    #replaces arg by name with values from the existing df
    def replace_args(self, event_data):
        #check args list for string vals which match keys in param dict
        self.args = self.perm_args.copy() #copy we'll actually pass to the function

        # print(self.function.__name__, self.perm_args)

        for (arg, val) in iteritems(self.args):
        #   print(arg, val)
        #
        #   if getattr(val, '__iter__', False):
        #     self.args[arg] = val
          try:
              if val in event_data.keys():
                self.args[arg] = event_data[val]
          except TypeError: #For when we have an array -- like passing in an array of time points
            pass
        # print ("--> ", self.args)


    def set_waveform(self, waveform_dict):
        self.input_wf = waveform_dict[self.input_waveform_name]

    #This will work for a Calculator or Transformer, not a DB Lookup? (which I've broken for now)
    def process(self):
        return self.function(self.input_wf, **self.args)

#Classes that wrap functional implementations of calculators or transformers
class Calculator(ProcessorBase):
  def __init__(self, function, output_name, args={},  input_waveform="waveform"):
      super().__init__(function, output_name=output_name, perm_args = args, input_waveform=input_waveform)

class Transformer(ProcessorBase):
  def __init__(self, function, output_waveform, args={}, input_waveform="waveform"):
      super().__init__(function, output_name=output_waveform, perm_args = args, input_waveform=input_waveform)

class DatabaseLookup(ProcessorBase):
  def __init__(self, function, args={}, output_name=None):
      print("Database Lookup has been murdered in cold blood.  B. Shanks, 8/15/18.  Either get it working or remove the DB call from your processor.")
      sys.exit()
#     self.function = function
#
#     self.output_name = output_name
#
#   def replace_args(self, param_dict):
#     #check args list for string vals which match keys in param dict
#     self.args = self.perm_args.copy() #copy we'll actually pass to the function
#
#     for (arg, val) in iteritems(self.args):
#       if val in param_dict.keys():
#         self.args[arg] = param_dict[val]
#
#   def process(self):
#     return self.function(**self.args)

# class Tier0Passer(ProcessorBase):
#   def __init__(self, t0_name, output_name=None):
#     self.t0_name = t0_name
#     if output_name is None: output_name = t0_name
#     self.output_name = output_name
#
#   def replace_args(self, param_dict):
#     self.t0_value = param_dict[self.t0_name]
#
#   def process(self):
#     return self.t0_value
