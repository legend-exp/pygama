import numpy as np
import pandas as pd


class WaveformFrame:
    """ do stuff to zipped dataframes of numpy arrays w. different wf types """

    def __init__(self, wf_df):
        pass
        # print(wf_df.shape, wf_df.columns)


class Waveform:
    """
    This is the pygama waveform class.
    Intended for operations on a properly processed pandas dataframe.
    """

    def __init__(self, wf_data, sample_period):
        '''
        sample_period is in ns
        '''
        self.data = wf_data.astype('float_')
        self.sample_period = sample_period
        self.amplitude = np.amax(self.data)

    def window_waveform(self,
                        time_point=0.5,
                        early_samples=200,
                        num_samples=400,
                        method="percent",
                        use_slope=False):
        """
        Windows waveform around a risetime percentage timepoint
        time_point: percentage (0-1)
        early_samples: samples to include before the calculated time_point
        num_samples: total number of samples to include
        """

        # don't mess with the original data
        wf_copy = np.copy(self.data)

        #bl subtract
        try:
            wf_copy = wf_copy - self.bl_int
            if use_slope:
                wf_copy = wf_copy - (np.arange(len(wf_copy)) * self.bl_slope)

        except AttributeError:
            p = fit_baseline(wf_copy)
            wf_copy = wf_copy - (p[1] + np.arange(len(wf_copy)) * p[0])

        #Normalize the waveform by the calculated energy (noise-robust amplitude estimation)
        if method == "percent":
            wf_norm = np.copy(wf_copy) / self.amplitude
            # tp_idx = np.int( calc_timepoint(wf_norm, time_point, doNorm=False  ))
            print("clint broke this")
        elif method == "value":
            tp_idx = np.argmax(wf_copy > time_point)
        else:
            raise ValueError

        # self.windowed_wf = center(wf_copy, tp_idx, early_samples, num_samples-early_samples)
        print("clint broke this too")
        self.window_length = num_samples

        return self.windowed_wf


class MultisampledWaveform(Waveform):
    """
    Multisampled WF class.
    """

    def __init__(self, time, wf_data, sample_period, full_sample_range, *args,
                 **kwargs):
        self.time = time
        self.full_sample_range = full_sample_range
        super().__init__(wf_data, sample_period, **kwargs)

    def get_waveform(self):
        return self.data[self.full_sample_range[0]:self.full_sample_range[-1]]
