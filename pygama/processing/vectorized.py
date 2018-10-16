""" -- vectorized pygama processors --
take big 2d blocks of waveforms (ndarrays)
with a waveform on every row, and apply
various transforms / calculations.
"""
import numpy as np
import pandas as pd
from abc import ABC

class VectorProcess:
    """ Handle vectorized calculators and transforms.
    Keep an internal 'intercom' of calculator results and waveform transforms.
    """
    def __init__(self, default_list=False):
        self.proc_list = []
        self.digitizer = None
        self.calc_df = None # df for calculation results (no wfs)
        self.wave_dict = {} # wfs only, NxM arrays (unpacked)
        if default_list:
            self.SetDefaultList()

    def Process(self, data_df, wfnames_out=None):
        """ Apply each processor to the Tier 0 input dataframe,
        and return a Tier 1 dataframe (i.e. gatified single-valued).
        Optionally return a dataframe with the waveform objects.
        """
        # save an ndarray of the raw waveforms
        wf_raw = [row for row in data_df["waveform"].values]
        self.wave_dict["waveform"] = np.array(wf_raw)

        # apply each processsor to the input dataframe
        for processor in self.proc_list:

            processor.set_block(data_df)
            p_result = processor.process(self.calc_df)

            if isinstance(processor, VectorCalculator):
                self.update_calcs(p_result)
                data_df = data_df.join(p_result, how='left')

            elif isinstance(processor, VectorTransformer):
                self.update_waveforms(p_result)

        if wfnames_out is not None:
            return data_df, self.zip_waveforms(wfnames_out)

        return data_df

    def AddCalculator(self, *args, **kwargs):
        self.proc_list.append(VectorCalculator(*args, **kwargs))

    def AddTransformer(self, *args, **kwargs):
        self.proc_list.append(VectorTransformer(*args, **kwargs))

    def SetDefaultList(self):
        self.AddCalculator(avg_baseline,
                          wf_names = ["waveform"],
                          fun_args = {"i_end":700})
        self.AddCalculator(fit_baseline,
                          wf_names = ["waveform"],
                          fun_args = {"i_end":700})
        self.AddTransformer(bl_subtract,
                           wf_names = ["waveform"],
                           fun_args = {"test":False})

    def update_calcs(self, result_df):
        """ update the internal dataframe of calculator results
        s/t other processors can access them
        """
        if self.calc_df is None:
            self.calc_df = result_df
        else:
            self.calc_df = self.calc_df.join(result_df, how='left')

    def update_waveforms(self, wf_dict):
        """ update the internal dictionary of NxM ndarray waveforms,
        taking output from one of the VectorTransformers
        """
        for wf in wf_dict:
            self.wave_dict[wf] = wf_dict[wf]

    def zip_waveforms(self, wfnames_out):
        """ embed ndarray waveform blocks into 1d DataFrame columns """
        wf_cols = {}
        for wf in wfnames_out:
            try:
                wf_block = self.wave_dict[wf]
            except KeyError:
                print("waveform type '{}' not available! exiting!".format(wf))

            wf_cols[wf] = [row for row in wf_block]

        return pd.DataFrame(wf_cols)


class VectorProcessorBase(ABC):

    def __init__(self, function, wf_names, fun_args={}):
        """ save some argunemnts specific to this transform/calculator """
        self.function = function
        self.wf_names = wf_names # name the wf type(s) we are operating on
        self.fun_args = fun_args # so fun

        # ndarrays for different wf types that this processor operates on
        self.block_dict = {} # usually just 'waveform'

    def set_block(self, data_df):
        """ convert an embedded Tier 0 list to a NxM df.
        remember, we can in principle operate on multiple wf types if needed
        """
        for wf_name in self.wf_names:
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

    def __init__(self, function, wf_names, fun_args={}):
        super().__init__(function, wf_names, fun_args)


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
    def __init__(self, function, wf_names, fun_args={}):
        super().__init__(function, wf_names, fun_args)


def bl_subtract(data_block, calc_df, test=False):
    """ Return an ndarray of baseline-subtracted waveforms
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

    if test:
        # alternate - transform based off avg_baseline calculator
        bl_avg = calc_df["bl_avg"].values[:,np.newaxis]
        blsub_avgs = wfs - bl_avg

        # quick diagnostic plot
        import matplotlib.pyplot as plt
        iwf = 1
        plt.plot(np.arange(nsamp), wfs[iwf], '-r', label="raw")
        plt.plot(np.arange(nsamp), blsub_wfs[iwf], '-b', label="bl_sub")
        plt.plot(np.arange(nsamp), blsub_avgs[iwf], '-g', label="bl_avg")
        plt.legend()
        plt.show()
        exit()

    return {"wf_blsub": blsub_wfs} # note, floats are gonna take up more memory