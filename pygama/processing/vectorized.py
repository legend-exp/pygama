"""
-- vectorized pygama processors --
take big 2d blocks of waveforms (ndarrays)
with a waveform on every row, and apply
various transforms / calculations.
"""
import numpy as np
import pandas as pd
from abc import ABC

class VectorProcess:
    """
    Handle vectorized calculators and transforms.
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
        # save an ndarray of the waveforms
        iwf = data_df.columns.get_loc(0) # wf sample 0
        self.wave_dict["waveform"] = data_df.iloc[:, iwf:].values

        # save the non-wf parts of the input dataframe separately
        self.calc_df = data_df.iloc[:, :iwf]

        # apply each processsor
        for processor in self.proc_list:
            # print(processor.function.__name__)

            p_result = processor.process(self.wave_dict, self.calc_df)

            if isinstance(processor, VectorCalculator):
                self.calc_df = self.calc_df.join(p_result, how='left')

            elif isinstance(processor, VectorTransformer):
                for wftype in p_result:
                    self.wave_dict[wftype] = p_result[wftype]

        if wfnames_out is not None:
            wf_out = {wf : self.wave_dict[wf] for wf in wfnames_out}
            return self.calc_df, wf_out

        return self.calc_df

    def AddCalculator(self, *args, **kwargs):
        self.proc_list.append(VectorCalculator(*args, **kwargs))

    def AddTransformer(self, *args, **kwargs):
        self.proc_list.append(VectorTransformer(*args, **kwargs))

    def SetDefaultList(self):
        self.AddCalculator(avg_baseline, fun_args = {"i_end":500})
        self.AddCalculator(fit_baseline, fun_args = {"i_end":500})
        self.AddTransformer(bl_subtract, fun_args = {"test":False})


class VectorProcessorBase(ABC):

    def __init__(self, function, fun_args={}):
        """ save some argunemnts specific to this transform/calculator """
        self.function = function
        self.fun_args = fun_args # so fun

    def process(self, wave_dict, calc_df):
        """ run the given calculation on the wf block in wave_dict.
        can also use results from other calculations via calc_df.
        individual processor functions can decide if they want to use the
        df axes, or convert to a numpy array for extra speed
        """
        return self.function(wave_dict, calc_df, **self.fun_args)

# ==============================================================================

class VectorCalculator(VectorProcessorBase):

    def __init__(self, function, fun_args={}):
        super().__init__(function, fun_args)


def avg_baseline(wave_dict, calc_df, i_start=0, i_end=500):
    """ Simple mean, vectorized version of baseline calculator """

    nd_block = wave_dict["waveform"]

    # find wf means
    avgs = np.mean(nd_block[:, i_start:i_end], axis=1)

    # return df with column names
    return pd.DataFrame(avgs, columns = ["bl_avg"])


def fit_baseline(wave_dict, calc_df, i_start=0, i_end=500, order=1):
    """ Polynomial fit [order], vectorized version of baseline calculator """

    nd_block = wave_dict["waveform"]

    # run polyfit
    x = np.arange(i_start, i_end)
    wfs = nd_block[:, i_start:i_end].T
    pol = np.polynomial.polynomial.polyfit(x, wfs, order).T

    # return df with column names
    return pd.DataFrame(pol, columns = ["bl_int", "bl_slope"])

# ==============================================================================

class VectorTransformer(VectorProcessorBase):
    def __init__(self, function, fun_args={}):
        super().__init__(function, fun_args)


def bl_subtract(wave_dict, calc_df, test=False):
    """ Return an ndarray of baseline-subtracted waveforms
    Depends on fit_baseline calculator.
    for reference, the non-vector version is just:
    return waveform - (bl_0 + bl_1 * np.arange(len(waveform)))
    """
    wfs = wave_dict["waveform"]
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