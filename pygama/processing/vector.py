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
        self.waves = {} # wfs only, NxM arrays (unpacked)

        if default_list:
            self.SetDefaultList()

    def Process(self, data_df, wfnames_out=None):
        """ Apply each processor to the Tier 0 input dataframe,
        and return a Tier 1 dataframe (i.e. gatified single-valued).
        Optionally return a dataframe with the waveform objects.
        """
        # save an ndarray of the waveforms
        iwf = data_df.columns.get_loc(0) # wf sample 0
        self.waves["waveform"] = data_df.iloc[:, iwf:].values

        # save the non-wf parts of the input dataframe separately
        self.calc_df = data_df.iloc[:, :iwf].copy() # make a new df here

        # apply each processsor
        for processor in self.proc_list:
            # print(processor.function.__name__)

            p_result = processor.process(self.waves, self.calc_df)

            if isinstance(processor, VectorCalculator):
                # calc_df is updated inside the functions
                pass

            elif isinstance(processor, VectorTransformer):
                for wftype in p_result:
                    self.waves[wftype] = p_result[wftype]

        if wfnames_out is not None:
            wf_out = {wf : self.waves[wf] for wf in wfnames_out}
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
        self.AddTransformer(trap_filter, fun_args = {"test":False})


class VectorProcessorBase(ABC):

    def __init__(self, function, fun_args={}):
        """ save some argunemnts specific to this transform/calculator """
        self.function = function
        self.fun_args = fun_args # so fun

    def process(self, waves, calc_df):
        """ run the given calculation on the wf block in waves.
        can also use results from other calculations via calc_df.
        individual processor functions can decide if they want to use the
        df axes, or convert to a numpy array for extra speed
        """
        return self.function(waves, calc_df, **self.fun_args)

# ==============================================================================

class VectorCalculator(VectorProcessorBase):

    def __init__(self, function, fun_args={}):
        super().__init__(function, fun_args)


def avg_baseline(waves, calc_df, i_start=0, i_end=500):
    """ Simple mean, vectorized version of baseline calculator """

    wf_block = waves["waveform"]

    # find wf means
    avgs = np.mean(wf_block[:, i_start:i_end], axis=1)

    # add the result as a new column
    calc_df["bl_avg"] = avgs
    return calc_df


def fit_baseline(waves, calc_df, i_start=0, i_end=500, order=1):
    """ Polynomial fit [order], vectorized version of baseline calculator
    TODO: arbitrary orders?
    """
    wf_block = waves["waveform"]

    # run polyfit
    x = np.arange(i_start, i_end)
    wfs = wf_block[:, i_start:i_end].T
    pol = np.polynomial.polynomial.polyfit(x, wfs, order).T

    # add the result as new columns
    calc_df["bl_int"] = pol[:,0]
    calc_df["bl_slope"] = pol[:,1]
    return calc_df


def trap_max(waves, calc_df, test=False):
    """ calculate maximum of trapezoid filter - no pride here """

    wfs = waves["wf_trap"]

    maxes = np.amax(wfs, axis=1)

    if test:
        import matplotlib.pyplot as plt
        iwf = 1
        plt.plot(np.arange(len(wfs[iwf])), wfs[iwf], '-r')
        plt.axhline(maxes[iwf])
        plt.show()
        exit()

    calc_df["trap_max"] = maxes
    return calc_df


# ==============================================================================

class VectorTransformer(VectorProcessorBase):
    def __init__(self, function, fun_args={}):
        super().__init__(function, fun_args)


def bl_subtract(waves, calc_df, test=False):
    """ Return an ndarray of baseline-subtracted waveforms
    Depends on fit_baseline calculator.
    for reference, the non-vector version is just:
    return waveform - (bl_0 + bl_1 * np.arange(len(waveform)))
    """
    wfs = waves["waveform"]
    nwfs, nsamp = wfs.shape[0], wfs.shape[1]

    bl_0 = calc_df["bl_int"].values[:,np.newaxis]

    slope_vals = calc_df["bl_slope"].values[:,np.newaxis]
    bl_1 = np.tile(np.arange(nsamp), (nwfs, 1)) * slope_vals

    # blsub_wfs = wfs - bl_0
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


def trap_filter(waves, calc_df, rt=400, ft=200, dt=0, test=False):

    wfs = waves["wf_blsub"]
    nwfs, nsamp = wfs.shape[0], wfs.shape[1]

    wfs_minus_ramp = np.zeros_like(wfs)
    wfs_minus_ramp[:, :rt] = 0
    wfs_minus_ramp[:, rt:] = wfs[:, :nsamp - rt]

    wfs_minus_ft_and_ramp = np.zeros_like(wfs)
    wfs_minus_ft_and_ramp[:, :(ft + rt)] = 0
    wfs_minus_ft_and_ramp[:, (ft + rt):] = wfs[:, :nsamp - ft - rt]

    wfs_minus_ft_and_2ramp = np.zeros_like(wfs)
    wfs_minus_ft_and_2ramp[:, :(ft + 2 * rt)] = 0
    wfs_minus_ft_and_2ramp[:, (ft + 2 * rt):] = wfs[:, :nsamp - ft - 2 * rt]

    scratch = wfs - (wfs_minus_ramp + wfs_minus_ft_and_ramp + wfs_minus_ft_and_2ramp)

    trap_wfs = np.zeros_like(wfs)
    trap_wfs = np.cumsum(trap_wfs + scratch, axis=1) / rt

    if test:
        # diagnostic plot
        import matplotlib.pyplot as plt
        import pygama.processing.transforms as pt
        iwf = 2
        plt.plot(np.arange(nsamp), wfs[iwf], '-r', label='raw')
        # plt.plot(np.arange(nsamp), wfs_minus_ramp[iwf], '-b', label='wf-ramp')
        # plt.plot(np.arange(nsamp), wfs_minus_ft_and_ramp[iwf], '-g', label='wf-ft-ramp')
        # plt.plot(np.arange(nsamp), wfs_minus_ft_and_2ramp[iwf], '-m', label='wf-ft-2ramp')
        # plt.plot(np.arange(nsamp), scratch[iwf], '-b', label='scratch')
        plt.plot(np.arange(nsamp), trap_wfs[iwf], '-g', lw=4, label='trap')

        trapwf = pt.trap_filter(wfs[iwf])
        plt.plot(np.arange(len(trapwf)), trapwf, '-k', label='bentrap')

        plt.ylim(-1000, 1.2*np.amax(wfs[iwf]))
        plt.legend()
        plt.show()
        exit()

    return {"wf_trap": trap_wfs}

