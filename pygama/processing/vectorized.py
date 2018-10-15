""" -- vectorized pygama processors --
take big 2d blocks of waveforms (ndarrays)
with a waveform on every row, and apply
various transforms / calculations.
"""
import numpy as np
import pandas as pd
from abc import ABC

class VectorizedProcessorList():
    """ Handle vectorized calculators and transforms. """

    def __init__(self):
        self.proc_list = []
        self.digitizer = None

    def Process(self, data_block):
        """ main routine to generate Tier 1 Dataframes.
        Apply each processor to the input dataframe,
        and return a new one """

        new_cols = {}

        for processor in self.proc_list:
            print("Applying processor:",processor.function)

            if isinstance(processor, VectorCalculator):

                processor.set_block(data_block)
                result_df = processor.process()

                for colname in processor.out_name:
                    new_cols[colname] = result_df[colname]

            if isinstance(processor, VectorTransformer):

                print("i'm in danger!")

        return pd.concat(new_cols, axis=1)


    def AddCalculator(self, *args, **kwargs):
        self.proc_list.append(VectorCalculator(*args, **kwargs))

    def AddTransformer(self, *args, **kwargs):
        self.proc_list.append(VectorTransformer(*args, **kwargs))


class VectorProcessorBase(ABC):

    def __init__(self, function, in_name, out_name, fun_args={}):
        """ save some argunemnts specific to this transform/calculator """
        self.function = function
        self.in_name = in_name
        self.out_name = out_name
        self.fun_args = fun_args # so fun
        self.block_dict = {} # blocks for different wf types

    def set_block(self, data_block):
        """ convert the embedded Tier 0 list to a NxM df
        what should we do about keeping column labels?
        remember, we can have multiple wf types (from transforms)
        """
        for wf_name in self.in_name:
            wflist = [row for row in data_block[wf_name].values]
            wf_df = pd.DataFrame(np.array(wflist))
            self.block_dict[wf_name] = wf_df

    def process(self):
        """ input the dataframe, and return one.
        individual processors can decide if they want to use the df axes,
        or convert to a numpy array, for speed
        """
        return self.function(self.block_dict, **self.fun_args)

# ==============================================================================

class VectorCalculator(VectorProcessorBase):

    def __init__(self, function, in_name, out_name, fun_args={}):
        super().__init__(function, in_name, out_name, fun_args)


def avg_baseline(data_block, start_index=0, end_index=500):
    """ Simple mean, vectorized version of baseline calculator """

    # convert to ndarray
    nd_block = data_block["waveform"].values

    # find wf means
    avgs = np.mean(nd_block[:, start_index:end_index], axis=1)

    # return df with column names
    return pd.DataFrame(avgs, columns = ["bl_avg"])


def fit_baseline(data_block, start_index=0, end_index=500, order=1):
    """ Polynomial fit [order], vectorized version of baseline calculator """

    # convert to ndarray
    nd_block = data_block["waveform"].values

    # run polyfit
    x = np.arange(start_index, end_index)
    wfs = nd_block[:, start_index:end_index].T
    pol = np.polynomial.polynomial.polyfit(x, wfs, order).T

    # return df with column names
    return pd.DataFrame(pol, columns = ["bl_int", "bl_slope"])

# ==============================================================================

class VectorTransformer(VectorProcessorBase):
    def __init__(self, function, in_name, out_name, fun_args={}):
        super().__init__(function, in_name, out_name, fun_args)
