import numpy as np
import pandas as pd
import sys
from scipy import signal
import itertools
import array

from .data_loading import DataLoader
from .waveform import Waveform


class Digitizer(DataLoader):
    """ handle any data loader which contains waveform data """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply_settings(self, settings):
        """ apply user settings specific to this card and run """

        if settings["digitizer"] == self.decoder_name:

            self.window = False
            sk = settings.keys()
            if "window" in sk:
                self.window = True
                self.win_type = settings["window"]
            if "n_samp" in sk:
                self.n_samp = settings["n_samp"]
            if "n_blsamp" in sk:
                self.n_blsamp = settings["n_blsamp"]


class Gretina4MDecoder(Digitizer):
    """ handle MJD Gretina digitizers """
    def __init__(self, *args, **kwargs):

        self.decoder_name = 'ORGretina4MWaveformDecoder'
        self.class_name = 'ORGretina4MModel'

        # store an entry for every event -- this is what we convert to pandas
        self.decoded_values = {
            "packet_id": [],
            "ievt": [],
            "energy": [],
            "timestamp": [],
            "channel": [],
            "board_id": [],
            "waveform": [],
        }
        super().__init__(*args, **kwargs) # also initializes the garbage df

        self.chan_list = None
        self.active_channels = self.find_active_channels()
        self.is_multisampled = True
        self.event_header_length = 18
        self.sample_period = 10  # ns
        self.gretina_event_no = 0
        self.window = False
        self.n_blsamp = 500
        self.ievt = 0

    def crate_card_chan(self, crate, card, channel):
        return (crate << 9) + (card << 4) + (channel)

    def find_active_channels(self):
        """ Only do this for multi-detector data """

        active_channels = []
        if self.df_metadata is None:
            return active_channels

        for index, row in self.df_metadata.iterrows():
            crate, card = index
            for chan, chan_en in enumerate(row.Enabled):
                if chan_en:
                    active_channels.append(
                        self.crate_card_chan(crate, card, chan))

        return active_channels

    def decode_event(self, event_data_bytes, packet_id, header_dict):
        """ Parse the header for an individual event """

        self.gretina_event_no += 1
        event_data = np.fromstring(event_data_bytes, dtype=np.uint16)
        card = event_data[1] & 0x1F
        crate = (event_data[1] >> 5) & 0xF
        channel = event_data[4] & 0xf
        board_id = (event_data[4] & 0xFFF0) >> 4
        timestamp = event_data[6] + (event_data[7] << 16) + (event_data[8] << 32)
        energy = event_data[9] + ((event_data[10] & 0x7FFF) << 16)
        wf_data = event_data[self.event_header_length:]

        ccc = self.crate_card_chan(crate, card, channel)
        if ccc not in self.active_channels:
            # should store this in a garbage data frame
            return

        # if the wf is too big for pytables, we can window it,
        # but we might get some garbage
        if self.window:
            wf = Waveform(wf_data, self.sample_period, self.decoder_name)
            waveform = wf.window_waveform(self.win_type,
                                          self.n_samp,
                                          self.n_blsamp,
                                          test=False)
            if wf.is_garbage:
                ievt = self.ievtg
                self.ievtg += 1
                self.garbage_count += 1

        if len(wf_data) > 2500 and self.h5_format == "table":
            print("WARNING: too many columns for tables output,",
                  "         reverting to saving as fixed hdf5 ...")
            self.h5_format = "fixed"

        waveform = wf_data.astype("int16")

        # set the event number (searchable HDF5 column)
        ievt = self.ievt
        self.ievt += 1

        # send any variable with a name in "decoded_values" to the pandas output
        self.format_data(locals())


class SIS3302Decoder(Digitizer):
    """ handle Struck 3302 digitizer """

    def __init__(self, *args, **kwargs):

        self.decoder_name = 'ORSIS3302DecoderForEnergy'
        self.class_name = 'ORSIS3302Model'

        # store an entry for every event -- this is what goes into pandas
        self.decoded_values = {
            "packet_id": [],
            "ievt": [],
            "energy": [],
            "energy_first": [],
            "timestamp": [],
            "channel": [],
            "ts_lo": [],
            "ts_hi": [],
            "waveform": [],
            # "energy_wf": []
        }
        super().__init__(*args, **kwargs) # also initializes the garbage df

        self.event_header_length = 1
        self.sample_period = 10  # ns
        self.h5_format = "table"
        self.n_blsamp = 2000
        self.ievt = 0
        self.ievtg = 0

    def decode_event(self,
                     event_data_bytes,
                     packet_id,
                     header_dict,
                     verbose=False):
        """
        see README for the 32-bit data word diagram
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

        # get the footer
        energy = evt_data_32[-4]
        energy_first = evt_data_32[-3]
        extra_flags = evt_data_32[-2]

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
                  (ievt, expected_wf_length, len(wf_data)))
            exit()

        # final raw wf array
        waveform = wf_data

        # if the wf is too big for pytables, we can window it,
        # but we might get some garbage
        if self.window:
            wf = Waveform(wf_data, self.sample_period, self.decoder_name)
            win_wf, win_ts = wf.window_waveform(self.win_type,
                                                self.n_samp,
                                                self.n_blsamp,
                                                test=False)
            ts_lo, ts_hi = win_ts[0], win_ts[-1]

            waveform = win_wf # modify final wf array

            if wf.is_garbage:
                ievt = self.ievtg
                self.ievtg += 1
                self.format_data(locals(), wf.is_garbage)
                return

        if len(waveform) > self.pytables_col_limit and self.h5_format == "table":
            print("WARNING: too many columns for tables output,\n",
                  "         reverting to saving as fixed hdf5 ...")
            self.h5_format = "fixed"

        # set the event number (searchable HDF5 column)
        ievt = self.ievt
        self.ievt += 1

        # send any variable with a name in "decoded_values" to the pandas output
        self.format_data(locals())
        
        
        
        
        
        
 
        
        
class SIS3316Decoder(Digitizer):
    """ handle Struck 3316 digitizer """

    def __init__(self, *args, **kwargs):
        pass
        
        
        
        
