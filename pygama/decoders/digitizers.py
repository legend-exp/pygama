import numpy as np
import pandas as pd
import sys
from scipy import signal
import itertools
import array

from .dataloading import DataLoader
from ..waveform import Waveform, MultisampledWaveform

__all__ = ['Digitizer', 'Gretina4MDecoder', 'SIS3302Decoder', 'get_digitizers']


def get_digitizers():
    return [sub() for sub in Digitizer.__subclasses__()]


class Digitizer(DataLoader):
    """
    members:
    - decode_event (base is in DataLoader (dataloading.py))
    - create_df (also in DataLoader)
    - parse_event_data
    - reconstruct_waveform
    """

    def __init__(self, *args, **kwargs):

        self.split_waveform = False
        self.chan_list = None  # list of channels to decode
        if self.split_waveform:
            self.hf5_type = "table"
        else:
            self.hf5_type = "fixed"

        super().__init__(*args, **kwargs)

    def decode_event(self, event_data_bytes, event_number, header_dict):
        pass

    def parse_event_data(self, event_data):
        if self.split_waveform:
            wf_data = self.reconstruct_waveform(event_data)
        else:
            wf_data = event_data["waveform"]

        return Waveform(wf_data.astype('float_'), self.sample_period)

    def reconstruct_waveform(self, event_data_row):
        waveform = []
        for i in itertools.count(0, 1):
            try:
                sample = event_data_row["waveform_{}".format(i)]
                waveform.append(sample)
            except KeyError:
                break
        return np.array(waveform)

    def create_df(self):
        """ Overloads DataLoader::create_df (in dataloading.py)
        for multisampled waveforms.  Should this be in Gretina4MDecoder?
        """
        if self.split_waveform:
            waveform_arr = self.decoded_values.pop("waveform")
            waveform_arr = np.array(waveform_arr, dtype="int16")

            for i in range(waveform_arr.shape[1]):
                self.decoded_values["waveform_{}".format(i)] = waveform_arr[:, i]

            df = pd.DataFrame.from_dict(self.decoded_values)

            return df

        else:
            return super(Digitizer, self).create_df()


class Gretina4MDecoder(Digitizer):
    """
    inherits from Digitizer and DataLoader

    can inspect all methods with:
    `import inspect, pygama`
    `gr = pygama.Gretina4MDecoder()`
    `inspect.getmembers(gr)`
    can show data members with `gr.__dict__`

    min_signal_thresh: multiplier on noise ampliude required to process a signal:
    helps avoid processing a ton of noise
    chanList: list of channels to process

    members:
      - load_object_info (also in DataLoader)
      - crate_card_chan
      - find_active_channels
      - decode_event (parses the header for an individual event)
      - format_data (format the values into a pandas-friendly format)
      - test_decode_event
      - parse_event_data (takes a pandas df row from a decoded event,
        returns an instance of class MultisampledWaveform)
    """

    def __init__(self, *args, **kwargs):

        self.decoder_name = 'ORGretina4MWaveformDecoder'  #ORGretina4M'
        self.class_name = 'ORGretina4MModel'

        super().__init__(*args, **kwargs)

        # store an entry for every event -- this is what we convert to pandas
        self.decoded_values = {
            "event_number": [],
            "energy": [],
            "timestamp": [],
            "channel": [],
            "board_id": [],
            "waveform": []
        }
        try:
            self.chan_list = kwargs.pop("chan_list")
        except KeyError:
            self.chan_list = None
        try:
            self.load_object_info(kwargs.pop("object_info"))
        except KeyError:
            pass
        try:
            self.correct_presum = kwargs.pop("correct_presum")
        except KeyError:
            self.correct_presum = True
            pass

        #The header length "should" be 32 -- 30 from gretina header, 2 from orca header
        #but the "reserved" from 16 on seems to be good baseline info, so lets try to use it
        self.event_header_length = 18
        self.wf_length = 2032  #TODO: This should probably be determined more rigidly
        self.sample_period = 10  #ns
        self.gretina_event_no = 0

    def load_object_info(self, object_info):
        super().load_object_info(object_info)
        self.active_channels = self.find_active_channels()

    def crate_card_chan(self, crate, card, channel):
        return (crate << 9) + (card << 4) + (channel)

    def find_active_channels(self):
        """ Only do this for multi-detector data """
        active_channels = []

        if self.object_info is None:
            return active_channels

        for index, row in self.object_info.iterrows():
            crate, card = index
            for chan, chan_en in enumerate(row.Enabled):
                if chan_en:
                    active_channels.append(
                        self.crate_card_chan(crate, card, chan))
        return active_channels

    def decode_event(self, event_data_bytes, event_number, header_dict):
        """
        Parse the header for an individual event
        """
        self.gretina_event_no += 1
        event_data = np.fromstring(event_data_bytes, dtype=np.uint16)

        # this is for a uint32
        # channel = event_data[1]&0xF
        # board_id = (event_data[1]&0xFFF0)>>4
        # timestamp = event_data[2] + ((event_data[3]&0xFFFF)<<32)
        # energy = ((event_data[3]&0xFFFF0000)>>16) + ((event_data[4]&0x7F)<<16)
        # wf_data = event_data[self.event_header_length:(self.event_header_length+self.wf_length)]

        # this is for a uint16
        card = event_data[1] & 0x1F
        crate = (event_data[1] >> 5) & 0xF
        channel = event_data[4] & 0xf
        board_id = (event_data[4] & 0xFFF0) >> 4

        timestamp = event_data[6] + (event_data[7] << 16) + (
            event_data[8] << 32)
        energy = event_data[9] + ((event_data[10] & 0x7FFF) << 16)
        wf_data = event_data[self.event_header_length:]
        # (self.event_header_length+self.wf_length)*2]
        waveform = wf_data.astype("int16")

        ccc = self.crate_card_chan(crate, card, channel)

        if ccc not in self.active_channels:
            #TODO: should store this to garbage data frame or something
            return None
            # raise ValueError("{} found data from channel {}, which is not in active channel list.".format(self.__class__.__name__, ccc))
        elif self.chan_list is not None and ccc not in self.chan_list:
            return None

        # if crate_card_chan not in board_id_map:
        #    board_id_map[crate_card_chan] = board_id
        # else:
        #    if not board_id_map[crate_card_chan] == board_id:
        #        print("WARNING: previously channel %d had board serial id %d, now it has id %d" % (crate_card_chan, board_id_map[crate_card_chan], board_id))

        # send any variable with a name in "decoded_values" to the pandas output
        self.format_data(locals())

    def parse_event_data(self, event_data):
        '''
        event_data is a pandas df row from a decoded event
        '''
        #TODO: doesn't handle full-time downsampling (channel_sum, channel_div)
        #TODO: adaptive presum handling could be done better
        #TODO: you're hosed if presum div is set to be the same as number of presum

        #cast wf to double
        wf_data = super().parse_event_data(event_data).data

        if not self.correct_presum:
            return Waveform(wf_data[-self.wf_length:], self.sample_period)
        else:
            #TODO: I fix the presumming by looking for a spike in the current with a windowed convolution
            #This slows down the decoding by almost x2.  We should try to do something faster
            #we save crate_card_chan (for historical reasons), so decode that
            event_chan = int(event_data['channel'])
            crate = event_chan >> 9
            card = (event_chan & 0x1f0) >> 4
            chan = event_chan & 0xf

            # print(7779311)
            # print((crate,card,chan))
            # print(type(self.object_info))
            # print(len(self.object_info))
            # print(self.object_info.columns)
            # print(self.object_info["Baseline Restore Enabled"][crate])
            # print(self.object_info["Baseline Restore Enabled"][crate][card])
            # print(self.object_info.iloc[(crate,card)])

            #Get the right digitizer information:
            card_info = self.object_info.iloc[crate]

            multirate_sum = 10 if card_info["Mrpsrt"][chan] == 3 else 2**(
                card_info["Mrpsrt"][chan] + 1)
            multirate_div = 2**card_info["Mrpsdv"][chan]
            ratio = multirate_sum / multirate_div
            # "channel_div": 2**card["Chpsdv"][channum],
            # "channel_sum": 10 if card["Chpsrt"][channum] == 3 else 2 **(card["Chpsrt"][channum]+1),

            prere_cnt = card_info["Prerecnt"][chan]
            postre_cnt = card_info["Postrecnt"][chan]
            ft_cnt = card_info["FtCnt"][chan]
            ms_start_offset = 0

            idx_ft_start_expected = len(wf_data) - ft_cnt - 1
            idx_bl_end_expected = len(wf_data) - prere_cnt - postre_cnt - ft_cnt

            filter_len = 10
            filter_win_mult = 1

            filter_window = np.ones(filter_len)
            filter_window[:int(filter_len / 2)] *= 1
            filter_window[int(filter_len / 2):] *= -1

            # def get_index(expected_idx):
            # #TODO: seems to be slower than doing the full convolution?
            #     wf_window_len = 10
            #     window_min = np.amax([0, expected_idx-wf_window_len])
            #     window_max = np.amin([expected_idx+wf_window_len, len(wf_data)])
            #     wf_window = wf_data[window_min:window_max]
            #     wf_data_cat = np.concatenate((np.ones(filter_win_mult*filter_len)*wf_window[0], wf_window, np.ones(filter_win_mult*filter_len)*wf_window[-1]))
            #     wf_diff = signal.convolve(wf_data_cat, filter_window, "same")
            #     idx_jump = np.argmax(np.abs(  wf_diff[filter_win_mult*filter_len:-filter_win_mult*filter_len])) + window_min
            #     return idx_jump
            #
            # idx_bl_end = get_index(idx_bl_end_expected)
            # idx_ft_start = get_index(idx_ft_start_expected)

            #TODO: doing the convolution on the whole window is unnecessarily slow
            wf_data_cat = np.concatenate(
                (np.ones(filter_win_mult * filter_len) * wf_data[0], wf_data,
                 np.ones(filter_win_mult * filter_len) * wf_data[-1]))
            wf_diff = signal.convolve(wf_data_cat, filter_window, "same")

            idx_bl_end = np.argmax(
                np.abs(wf_diff[filter_win_mult * filter_len:-filter_win_mult *
                               filter_len][:idx_bl_end_expected + 4]))
            idx_ft_start = np.argmax(
                np.abs(wf_diff[filter_win_mult * filter_len:-filter_win_mult *
                               filter_len][idx_ft_start_expected -
                                           5:])) + idx_ft_start_expected - 5

            if (idx_bl_end < idx_bl_end_expected - 8) or (
                    idx_bl_end > idx_bl_end_expected + 2):
                #baseline is probably very near zero, s.t. its hard to see the jump.  just assume it where its meant to be.
                idx_bl_end = idx_bl_end_expected
            if (idx_ft_start < idx_ft_start_expected - 2) or (
                    idx_ft_start > idx_ft_start_expected):
                idx_ft_start = idx_ft_start_expected
                # raise ValueError()

            wf_data[:idx_bl_end] /= (ratio)
            wf_data[idx_ft_start:] /= (ratio)

            time_pre = np.arange(
                idx_bl_end) * self.sample_period * multirate_sum - (
                    len(wf_data) - self.wf_length)
            time_full = np.arange(
                idx_ft_start - idx_bl_end) * self.sample_period + time_pre[
                    -1] + self.sample_period  # + ms_start_offset)
            time_ft = np.arange(
                ft_cnt + 1
            ) * self.sample_period * multirate_sum + time_full[-1] + 0.5 * (
                self.sample_period * multirate_sum + ms_start_offset)

            time = np.concatenate((time_pre, time_full, time_ft))

            return MultisampledWaveform(
                time[-self.wf_length:], wf_data[-self.wf_length:],
                self.sample_period, [idx_bl_end, idx_ft_start])


class SIS3302Decoder(Digitizer):

    def __init__(self, *args, **kwargs):

        self.decoder_name = 'ORSIS3302DecoderForEnergy'
        self.class_name = 'ORSIS3302Model'
        self.event_header_length = 1
        self.sample_period = 10  #ns

        super().__init__(*args, **kwargs)

        # store an entry for every event -- this is what goes into pandas
        self.decoded_values = {
            "energy": [],
            "energy_first": [],
            "timestamp": [],
            "channel": [],
            "event_number": [],
            "waveform": [],
            "energy_wf": []
        }


    def get_name(self):
        return self.decoder_name

    def decode_event(self,
                     event_data_bytes,
                     event_number,
                     header_dict,
                     verbose=False):
        """
        # The SIS3302 can produce a waveform from two sources:
        #     1: ADC raw data buffer: This is a normal digitizer waveform
        #     2: Energy data buffer: Not sure what this is
        # Additionally, the ADC raw data buffer can take data in a buffer wrap mode, which seems
        # to cyclically fill a spot on memory, and it requires that you have to re-order the records
        # afterwards.
        # The details of how this header is formatted apparently wasn't important enough for the
        # SIS engineers to want to put it in the manual, so this is a guess in some places
        #
        #   0   xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx ORCA:
        #       ^^^^ ^^^- ---- ---- ---- ---- ---- ---- most sig bits of num records lost
        #       ---- ---- ---- ---- ---- ---- ^^^^ ^^^- least sig bits of num records lost
        #               ^ ^^^- ---- ---- ---- ---- ---- crate
        #                    ^ ^^^^ ---- ---- ---- ---- card
        #                           ^^^^ ^^^^ ---- ---- channel
        #                                             ^ buffer wrap mode
        #   1   xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx ORCA: length of waveform (longs)
        #   2   xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx ORCA: length of energy   (longs)
        #   3   xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx ORCA:
        #       ^^^^ ^^^^ ^^^^ ^^^^ ---- ---- ---- ---- timestamp[47:32]
        #                           ^^^^ ^^^^ ^^^^ ^^^^ "event header and ADC ID"
        #   4   xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx SIS:
        #       ^^^^ ^^^^ ^^^^ ^^^^ ---- ---- ---- ---- timestamp[31:16]
        #                           ^^^^ ^^^^ ^^^^ ^^^^ timestamp[15:0]
        #
        #       If the buffer wrap mode is enabled, there are two more words of header:
        # (5)   xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx ADC raw data length (longs)
        # (6)   xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx ADC raw data start index (longs)
        #
        #       After this, it will go into the two data buffers directly.
        #       These buffers are packed 16-bit words, like so:
        #
        #       xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx
        #       ^^^^ ^^^^ ^^^^ ^^^^ ---- ---- ---- ---- sample N + 1
        #                           ^^^^ ^^^^ ^^^^ ^^^^ sample N
        #
        #       here's an example of combining the 16 bit ints to get a 32 bit one.
        #       print(hex(evt_data_16[-1] << 16 | evt_data_16[-2]))
        #
        #       The first data buffer is the ADC raw data buffer, which is the usual waveform
        #       The second is the energy data buffer, which might be the output of the energy filter
        #       This code should handle arbitrary sizes of both buffers.
        #
        #       An additional complexity arises if buffer wrap mode is enabled.
        #       This apparently means the start of the buffer can be anywhere in the buffer, and
        #       it must be read circularly from that point. Not sure why it is done that way, but
        #       this should work correctly to disentagle that.
        #
        #       Finally, there should be a footer of 4 long words at the end:
        #  -4   xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx Energy max value
        #  -3   xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx Energy value from first value of energy gate
        #  -2   xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx This word is said to contain "pileup flag, retrigger flag, and trigger counter" in no specified locations...
        #  -1   1101 1110 1010 1101 1011 1110 1110 1111 Last word is always 0xDEADBEEF
        """

        # parse the raw event data into numpy arrays of 16 and 32 bit ints
        evt_data_32 = np.fromstring(event_data_bytes, dtype=np.uint32)
        evt_data_16 = np.fromstring(event_data_bytes, dtype=np.uint16)

        # start reading the binary, baby
        n_lost_msb = (evt_data_32[0] >> 25) & 0x7F
        n_lost_lsb = (evt_data_32[0] >> 2) & 0x7F
        n_lost_records = (n_lost_msb << 7) + n_lost_lsb
        crate = (evt_data_32[0] >> 21) & 0xF
        card = (evt_data_32[0] >> 16) & 0x1F
        channel = (evt_data_32[0] >> 8) & 0xFF
        buffer_wrap = evt_data_32[0] & 0x1
        crate_card_chan = (crate << 9) + (card << 4) + channel
        wf_length_32 = evt_data_32[1]
        ene_wf_length = evt_data_32[2]
        evt_header_id = evt_data_32[3] & 0xFF
        timestamp = evt_data_32[4] + ((evt_data_32[3] >> 16) & 0xFFFF)
        last_word = evt_data_32[-1]

        # compute expected and actual array dimensions
        wf_length16 = 2 * wf_length_32
        orca_header_length16 = 2
        sis_header_length16 = 12 if buffer_wrap else 8
        header_length16 = orca_header_length16 + sis_header_length16
        ene_wf_length16 = 2 * ene_wf_length
        footer_length16 = 8
        expected_wf_length = len(evt_data_16) - orca_header_length16 - sis_header_length16 - \
            footer_length16 - ene_wf_length16

        # error check: waveform size must match expectations
        if wf_length16 != expected_wf_length or last_word != 0xdeadbeef:
            print(len(evt_data_16), orca_header_length16, sis_header_length16,
                  footer_length16)
            print("ERROR: Waveform size %d doesn't match expected size %d." %
                  (wf_length16, expected_wf_length))
            print("       The Last Word (should be 0xdeadbeef):",
                  hex(last_word))
            exit()

        # indexes of stuff (all referring to the 16 bit array)
        i_wf_start = header_length16
        i_wf_stop = i_wf_start + wf_length16
        i_ene_start = i_wf_stop + 1
        i_ene_stop = i_ene_start + ene_wf_length16
        if buffer_wrap:
            # start somewhere in the middle of the record
            i_start_1 = evt_data_32[6] + header_length16 + 1
            i_stop_1 = i_wf_stop  # end of the wf record
            i_start_2 = i_wf_start  # beginning of the wf record
            i_stop_2 = i_start_1

        # handle the waveform(s)
        energy_wf = np.zeros(ene_wf_length16)  # not used rn
        if wf_length_32 > 0:
            if not buffer_wrap:
                wf_data = evt_data_16[i_wf_start:i_wf_stop]
            else:
                wf_data1 = evt_data_16[i_start_1:i_stop_1]
                wf_data2 = evt_data_16[i_start_2:i_stop_2]
                wf_data = np.concatenate([wf_data1, wf_data2])

        if len(wf_data) != expected_wf_length:
            print("ERROR: event %d, we expected %d WF samples and only got %d" %
                  (event_number, expected_wf_length, len(wf_data)))
            exit()
        waveform = wf_data.astype("int16")

        # get the footer
        energy = evt_data_32[-4]
        energy_first = evt_data_32[-3]
        extra_flags = evt_data_32[-2]

        # send any variable with a name in "decoded_values" to the pandas output
        self.format_data(locals())
