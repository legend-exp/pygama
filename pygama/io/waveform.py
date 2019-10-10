import sys
import numpy as np
import pandas as pd

class Waveform:

    def __init__(self, wf_data, sample_period, decoder_name=None):

        self.data = wf_data.astype('float_')
        self.sample_period = sample_period
        self.ns = len(self.data)
        self.ts = np.arange(0, self.ns * self.sample_period, self.sample_period)
        self.is_garbage = False

        # may want to handle digitizer-specific options, like multisampling
        # only mulitsample if a flag is set?
        # self.dig_types = ["ORGretina4MWaveformDecoder",
        #                    "ORSIS3302DecoderForEnergy"]

    def window_waveform(self, win_type, n_samp, n_blsamp, test=False):
        """ zoom in on the rising edge of a waveform,
        either by maximum value, or a percentage of the rising edge """

        # center around max value (fast)
        if win_type == "max":
            i_ctr = np.argmax(self.data)

        # baseline subtract and center around 50 percent timepoint
        # note: this doesn't work when the 50% value is within the
        # gaussian noise envelope, and we declare the evt garbage
        elif win_type == "tp":

            order = 1
            bl_ts = self.ts[0:n_blsamp]
            bl_wf = self.data[0:n_blsamp]
            bl_slope, bl_int = np.polyfit(bl_ts, bl_wf, order)
            bl_curve = bl_slope * bl_ts + bl_int
            blsub_wf = self.data - (bl_slope * self.ts + bl_int)

            # get index closest to the 50 percent timepoint
            wf_max = np.amax(blsub_wf)
            i_ctr = np.where(blsub_wf >= wf_max * .50)[0][0]

        # set the window indexes
        i_upper = i_ctr + n_samp // 2
        if i_upper > self.ns:
            i_upper = self.ns - 1

        ilower = i_ctr - n_samp // 2
        if ilower < 0:
            ilower = 0

        # window the waveform and timestamps
        win_wf = self.data[ilower:i_upper]
        win_ts = self.ts[ilower:i_upper]

        # declare the waveform garbage if necessary
        if len(win_wf) != n_samp:
            # print("ERROR", i_upper-ilower, len(win_wf))
            # test = True
            self.is_garbage = True
            # sys.exit()

        if test:
            # quick diagnostic
            import matplotlib.pyplot as plt
            plt.plot(self.ts, self.data, '-r')
            # plt.plot(bl_ts, bl_curve, '-b')
            # plt.plot(self.ts, blsub_wf, '-r')
            # plt.axvline(i_ctr * self.sample_period)
            # plt.plot(win_ts, win_wf - bl_int, '-b') # this is nice
            plt.plot(win_ts, win_wf, '-b')
            plt.show()
            sys.exit()

        return win_wf, win_ts


class WaveformMS(Waveform):
    """ multisampled waveform """

    def __init__(self):
        self.presum_idx = None