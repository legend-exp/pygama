import numpy as np
import pandas as pd

class Waveform:
    """
    This is the pygama waveform class.
    Intended for operations on a properly processed pandas dataframe.
    """
    def __init__(self, wf_data, sample_period):
        """ sample_period is in ns """
        self.data = wf_data.astype('float_')
        self.sample_period = sample_period
        self.amplitude = np.amax(self.data)
        print("HYYYEEEE")

    def window_waveform(self, time_point=0.5, early_samples=200,
                        num_samples=400, method="percent", use_slope=False):
        """
        Windows waveform around a risetime percentage timepoint
        time_point: percentage (0-1)
        early_samples: samples to include before the calculated time_point
        num_samples: total number of samples to include
        """
        print("I'M HERE!")

        # i guess window it symmetrically around the most extreme (+ or -) value

        sys.exit()

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

    # def parse_event_data(self, event_data):
    #     """ TODO: update/rewrite this function """
    #
    #     print("I'M HERE")
    #
    #     # cast wf to double
    #     wf_data = super().parse_event_data(event_data).data
    #
    #     if not self.is_multisampled:
    #         return Waveform(wf_data[-self.wf_length:], self.sample_period)
    #     else:
    #         #TODO: I fix the presumming by looking for a spike in the current with a windowed convolution
    #         #This slows down the decoding by almost x2.  We should try to do something faster
    #         #we save crate_card_chan (for historical reasons), so decode that
    #         event_chan = int(event_data['channel'])
    #         crate = event_chan >> 9
    #         card = (event_chan & 0x1f0) >> 4
    #         chan = event_chan & 0xf
    #
    #         # print(7779311)
    #         # print((crate,card,chan))
    #         # print(type(self.object_info))
    #         # print(len(self.object_info))
    #         # print(self.object_info.columns)
    #         # print(self.object_info["Baseline Restore Enabled"][crate])
    #         # print(self.object_info["Baseline Restore Enabled"][crate][card])
    #         # print(self.object_info.iloc[(crate,card)])
    #
    #         #Get the right digitizer information:
    #         card_info = self.object_info.iloc[crate]
    #
    #         multirate_sum = 10 if card_info["Mrpsrt"][chan] == 3 else 2**(
    #             card_info["Mrpsrt"][chan] + 1)
    #         multirate_div = 2**card_info["Mrpsdv"][chan]
    #         ratio = multirate_sum / multirate_div
    #         # "channel_div": 2**card["Chpsdv"][channum],
    #         # "channel_sum": 10 if card["Chpsrt"][channum] == 3 else 2 **(card["Chpsrt"][channum]+1),
    #
    #         prere_cnt = card_info["Prerecnt"][chan]
    #         postre_cnt = card_info["Postrecnt"][chan]
    #         ft_cnt = card_info["FtCnt"][chan]
    #         ms_start_offset = 0
    #
    #         idx_ft_start_expected = len(wf_data) - ft_cnt - 1
    #         idx_bl_end_expected = len(wf_data) - prere_cnt - postre_cnt - ft_cnt
    #
    #         filter_len = 10
    #         filter_win_mult = 1
    #
    #         filter_window = np.ones(filter_len)
    #         filter_window[:int(filter_len / 2)] *= 1
    #         filter_window[int(filter_len / 2):] *= -1
    #
    #         # def get_index(expected_idx):
    #         # #TODO: seems to be slower than doing the full convolution?
    #         #     wf_window_len = 10
    #         #     window_min = np.amax([0, expected_idx-wf_window_len])
    #         #     window_max = np.amin([expected_idx+wf_window_len, len(wf_data)])
    #         #     wf_window = wf_data[window_min:window_max]
    #         #     wf_data_cat = np.concatenate((np.ones(filter_win_mult*filter_len)*wf_window[0], wf_window, np.ones(filter_win_mult*filter_len)*wf_window[-1]))
    #         #     wf_diff = signal.convolve(wf_data_cat, filter_window, "same")
    #         #     idx_jump = np.argmax(np.abs(  wf_diff[filter_win_mult*filter_len:-filter_win_mult*filter_len])) + window_min
    #         #     return idx_jump
    #         #
    #         # idx_bl_end = get_index(idx_bl_end_expected)
    #         # idx_ft_start = get_index(idx_ft_start_expected)
    #
    #         #TODO: doing the convolution on the whole window is unnecessarily slow
    #         wf_data_cat = np.concatenate(
    #             (np.ones(filter_win_mult * filter_len) * wf_data[0], wf_data,
    #              np.ones(filter_win_mult * filter_len) * wf_data[-1]))
    #         wf_diff = signal.convolve(wf_data_cat, filter_window, "same")
    #
    #         idx_bl_end = np.argmax(
    #             np.abs(wf_diff[filter_win_mult * filter_len:-filter_win_mult *
    #                            filter_len][:idx_bl_end_expected + 4]))
    #         idx_ft_start = np.argmax(
    #             np.abs(wf_diff[filter_win_mult * filter_len:-filter_win_mult *
    #                            filter_len][idx_ft_start_expected -
    #                                        5:])) + idx_ft_start_expected - 5
    #
    #         if (idx_bl_end < idx_bl_end_expected - 8) or (
    #                 idx_bl_end > idx_bl_end_expected + 2):
    #             #baseline is probably very near zero, s.t. its hard to see the jump.  just assume it where its meant to be.
    #             idx_bl_end = idx_bl_end_expected
    #         if (idx_ft_start < idx_ft_start_expected - 2) or (
    #                 idx_ft_start > idx_ft_start_expected):
    #             idx_ft_start = idx_ft_start_expected
    #             # raise ValueError()
    #
    #         wf_data[:idx_bl_end] /= (ratio)
    #         wf_data[idx_ft_start:] /= (ratio)
    #
    #         time_pre = np.arange(
    #             idx_bl_end) * self.sample_period * multirate_sum - (
    #                 len(wf_data) - self.wf_length)
    #         time_full = np.arange(
    #             idx_ft_start - idx_bl_end) * self.sample_period + time_pre[
    #                 -1] + self.sample_period  # + ms_start_offset)
    #         time_ft = np.arange(
    #             ft_cnt + 1
    #         ) * self.sample_period * multirate_sum + time_full[-1] + 0.5 * (
    #             self.sample_period * multirate_sum + ms_start_offset)
    #
    #         time = np.concatenate((time_pre, time_full, time_ft))
    #
    #         return MultisampledWaveform(
    #             time[-self.wf_length:], wf_data[-self.wf_length:],
    #             self.sample_period, [idx_bl_end, idx_ft_start])





# class MultisampledWaveform(Waveform):
#     """
#     Multisampled WF class.
#     """
#     def __init__(self, time, wf_data, sample_period, full_sample_range, *args,
#                  **kwargs):
#         self.time = time
#         self.full_sample_range = full_sample_range
#         super().__init__(wf_data, sample_period, **kwargs)
#
#     def get_waveform(self):
#         return self.data[self.full_sample_range[0]:self.full_sample_range[-1]]
