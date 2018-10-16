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
        self.calc_df = None # df for calculation results (no wfs)

    def Process(self, data_df):
        """ main routine to generate Tier 1 Dataframes.
        Apply each processor to the input dataframe,
        and return a new one """

        for processor in self.proc_list:
            print("Applying processor:",processor.function.__name__)

            processor.set_block(data_df)
            result_df = processor.process(self.calc_df)

            if isinstance(processor, VectorCalculator):
                self.update_calcs(result_df)
                data_df = data_df.join(result_df, how='left')

            elif isinstance(processor, VectorTransformer):

                print(result_df.shape)

                # do i need like a zip/unzip wfs function into the nested arrays?


        return data_df

    def update_calcs(self, result_df):
        if self.calc_df is None:
            self.calc_df = result_df
        else:
            self.calc_df = self.calc_df.join(result_df, how='left')

    def AddCalculator(self, *args, **kwargs):
        self.proc_list.append(VectorCalculator(*args, **kwargs))

    def AddTransformer(self, *args, **kwargs):
        self.proc_list.append(VectorTransformer(*args, **kwargs))


class VectorProcessorBase(ABC):

    def __init__(self, function, wf_name, fun_args={}):
        """ save some argunemnts specific to this transform/calculator """
        self.function = function
        self.wf_name = wf_name # name the wf type(s) we are operating on
        self.fun_args = fun_args # so fun

        # ndarray blocks for different wf types
        # should this be here?  don't we want other transforms to be able
        # to operate on the same numpy matrices, to avoid copying?
        self.block_dict = {}

    def set_block(self, data_df):
        """ convert an embedded Tier 0 list to a NxM df.
        remember, we can in principle operate on multiple wf types (from transforms)
        but usually self.wf_name will just be 'waveform'
        """
        for wf_name in self.wf_name:
            wflist = [row for row in data_df[wf_name].values]
            wf_df = pd.DataFrame(np.array(wflist))
            self.block_dict[wf_name] = wf_df

    def process(self, calc_df):
        """ run the given calculation on the wf block in self.block_dict.
        can also use results from other calculations via calc_df.
        individual processor functions can decide if they want to use the
        df axes, or convert to a numpy array for extra speed
        """
        return self.function(self.block_dict, calc_df, **self.fun_args)

# ==============================================================================

class VectorCalculator(VectorProcessorBase):

    def __init__(self, function, wf_name, fun_args={}):
        super().__init__(function, wf_name, fun_args)


def avg_baseline(data_block, calc_df, i_start=0, i_end=500):
    """ Simple mean, vectorized version of baseline calculator """

    # convert to ndarray
    nd_block = data_block["waveform"].values

    # find wf means
    avgs = np.mean(nd_block[:, i_start:i_end], axis=1)

    # return df with column names
    return pd.DataFrame(avgs, columns = ["bl_avg"])


def fit_baseline(data_block, calc_df, i_start=0, i_end=500, order=1):
    """ Polynomial fit [order], vectorized version of baseline calculator """

    # convert to ndarray
    nd_block = data_block["waveform"].values

    # run polyfit
    x = np.arange(i_start, i_end)
    wfs = nd_block[:, i_start:i_end].T
    pol = np.polynomial.polynomial.polyfit(x, wfs, order).T

    # return df with column names
    return pd.DataFrame(pol, columns = ["bl_int", "bl_slope"])

# ==============================================================================

class VectorTransformer(VectorProcessorBase):
    def __init__(self, function, wf_name, fun_args={}):
        super().__init__(function, wf_name, fun_args)


def bl_subtract(data_block, calc_df):
    """ Return an NxM ndarray of baseline-subtracted waveforms
    Depends on fit_baseline calculator.
    for reference, the non-vector version is just:
    return waveform - (bl_0 + bl_1 * np.arange(len(waveform)))
    """
    wfs = data_block["waveform"].values
    nwfs, nsamp = wfs.shape[0], wfs.shape[1]

    bl_0 = calc_df["bl_int"].values[:,np.newaxis]

    slope_vals = calc_df["bl_slope"].values[:,np.newaxis]
    bl_1 = np.tile(np.arange(nsamp), (nwfs, 1)) * slope_vals

    blsub_wfs = wfs - (bl_0 + bl_1)

    # alternate - transform based off avg_baseline calculator
    # bl_avg = calc_df["bl_avg"].values[:,np.newaxis]
    # blsub_avgs = wfs - bl_avg

    # # quick diagnostic
    import matplotlib.pyplot as plt
    iwf = 1
    plt.plot(np.arange(nsamp), wfs[iwf], '-r', label="raw")
    plt.plot(np.arange(nsamp), blsub_wfs[iwf], '-b', label="bl_sub")
    plt.plot(np.arange(nsamp), blsub_avgs[iwf], '-g', label="bl_avg")
    plt.legend()
    plt.show()

    return blsub_wfs