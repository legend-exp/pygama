import sys
import array
import itertools
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from pprint import pprint

from .io_base import DataTaker
from .waveform import Waveform

"""
FIXME:
these variables should be set by config if digitizer:
self.[window, win_type, n_samp, n_blsamp]

TODO:
Remove windowing feature completely, it's unnecessary with lh5 var-length arrs
"""

class ORCAStruck3302(DataTaker):
    """ 
    decode ORCA Struck 3302 digitizer data
    """
    def __init__(self, *args, **kwargs):

        self.decoder_name = 'ORSIS3302DecoderForEnergy'
        self.class_name = 'ORSIS3302Model'

        # store an entry for every event
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
        self.ievt_gbg = 0
        self.pytables_col_limit = 3000
        self.df_metadata = None # hack, this probably isn't right
        

    def decode_event(self, event_data_bytes, packet_id, header_dict, verbose=False):
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
        orca_helper_length16 = 2
        sis_header_length16 = 12 if buffer_wrap else 8
        header_length16 = orca_helper_length16 + sis_header_length16
        ene_wf_length16 = 2 * ene_wf_length
        footer_length16 = 8
        expected_wf_length = len(evt_data_16) - orca_helper_length16 - sis_header_length16 - \
            footer_length16 - ene_wf_length16

        # error check: waveform size must match expectations
        if wf_length16 != expected_wf_length or last_word != 0xdeadbeef:
            print(len(evt_data_16), orca_helper_length16, sis_header_length16,
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

        # # if the wf is too big for pytables, we can window it
        # if self.window:
        #     wf = Waveform(wf_data, self.sample_period, self.decoder_name)
        #     win_wf, win_ts = wf.window_waveform(self.win_type,
        #                                         self.n_samp,
        #                                         self.n_blsamp,
        #                                         test=False)
        #     ts_lo, ts_hi = win_ts[0], win_ts[-1]
        # 
        #     waveform = win_wf # modify final wf array
        # 
        #     if wf.is_garbage:
        #         ievt = self.ievt_gbg
        #         self.ievt_gbg += 1
        #         self.format_data(locals(), wf.is_garbage)
        #         return

        if len(waveform) > self.pytables_col_limit and self.h5_format == "table":
            print("WARNING: too many columns for tables output,\n",
                  "         reverting to saving as fixed hdf5 ...")
            self.h5_format = "fixed"

        # set the event number (searchable HDF5 column)
        ievt = self.ievt
        self.ievt += 1

        # send any variable with a name in "decoded_values" to the pandas output
        self.format_data(locals())

        
class LLAMAStruck3316(DataTaker):
    """ 
    decode Struck 3316 digitizer data
    
    TODO:
    handle per-channel data (gain, ...)
    most metadata of Struck header (energy, ...)
    """
    def __init__(self, metadata=None, *args, **kwargs):
        self.decoder_name = 'SIS3316Decoder'
        self.class_name = 'SIS3316'

        # store an entry for every event
        self.decoded_values = {
            "packet_id": [],
            "ievt": [],
            "energy_first": [],
            "energy": [],
            "timestamp": [],
            "peakhigh_index": [],
            "peakhigh_value": [],
            "information": [],
            "accumulator1": [],
            "accumulator2": [],
            "accumulator3": [],
            "accumulator4": [],
            "accumulator5": [],
            "accumulator6": [],
            "accumulator7": [],
            "accumulator8": [],
            "mawMax": [],
            "maw_before": [],
            "maw_after": [],
            "fadcID": [],
            "channel": [],
            "waveform": [],
        }

        self.config_names = []  #TODO at some point we want the metainfo here
        self.file_config = {}
        self.lh5_spec = {}
        self.file_config = self.readMetadata(metadata)
        print("We have {} adcs and {} samples per WF.".format(self.file_config["nadcs"],self.file_config["nsamples"]))

        super().__init__(*args, **kwargs) # also initializes the garbage df (whatever that means...)

        # self.event_header_length = 1 #?
        self.sample_period = 0  # ns, I will set this later, according to header info
        self.gain = 0           
        self.h5_format = "table"	#was table
        #self.n_blsamp = 2000
        self.ievt = 0       #event number
        self.ievt_gbg = 0      #garbage event number
        self.window = False
        self.df_metadata = metadata #seems that was passed to superclass before, try now like this
        self.pytables_col_limit = 3000

    def readMetadata(self, meta):
        nsamples = -1
        totChan = 0
        configs = {}
        adcOff = {}
        for fadc in meta:
            adcOff[fadc] = {}
            for channel in meta[fadc]:
                if nsamples == -1:
                    # FIXME everything is fixed to 1st existing channel.
                    nsamples = meta[fadc][channel]["SampleLength"]
                    configs["14BitFlag"] = meta[fadc][channel]["14BitFlag"]
                    #configs["ADCOffset"] = meta[fadc][channel]["ADCOffset"]
                    configs["FormatBits"] = meta[fadc][channel]["FormatBits"]
                    configs["Gain"] = meta[fadc][channel]["Gain"]
                    configs["SampleFreq"] = meta[fadc][channel]["SampleFreq"]
                    configs["SampleOffset"] = meta[fadc][channel]["SampleOffset"]
                    adcOff[fadc][channel] = meta[fadc][channel]["ADCOffset"]
                elif nsamples != meta[fadc][channel]["SampleLength"]:
                    print("samples not uniform!!!")
                totChan += 1
        configs["nadcs"] = totChan
        configs["nsamples"] = nsamples
        return configs
        
    def initialize(self, sample_period, gain):
        """
        sets certain global values from a run, like:
        sample_period: time difference btw 2 samples in ns
        gain: multiply the integer sample value with the gain to get the voltage in V
        Method has to be called before the actual decoding work starts !
        """
        self.sample_period = sample_period
        self.gain = gain
        
        
    def decode_event(self, event_data_bytes, packet_id, header_dict, fadcIndex, 
                     channelIndex, verbose=False):
        """
        see the llamaDAQ documentation for data word diagrams
        """
        
        if self.sample_period == 0:
            print("ERROR: Sample period not set; use initialize() before using decode_event() on SIS3316Decoder")
            raise Exception ("Sample period not set")
        
        # parse the raw event data into numpy arrays of 16 and 32 bit ints
        evt_data_32 = np.fromstring(event_data_bytes, dtype=np.uint32)
        evt_data_16 = np.fromstring(event_data_bytes, dtype=np.uint16)
        
        # e sti gran binaries non ce li metti
        timestamp = ((evt_data_32[0] & 0xffff0000) << 16) + evt_data_32[1]
        format_bits = (evt_data_32[0]) & 0x0000000f
        offset = 2
        if format_bits & 0x1:
            peakhigh_value = evt_data_16[4]
            peakhigh_index = evt_data_16[5]
            information = (evt_data_32[offset+1] >> 24) & 0xff
            accumulator1 = evt_data_32[offset+2]
            accumulator2 = evt_data_32[offset+3]
            accumulator3 = evt_data_32[offset+4]
            accumulator4 = evt_data_32[offset+5]
            accumulator5 = evt_data_32[offset+6]
            accumulator6 = evt_data_32[offset+7]
            offset += 7
        else:
            peakhigh_value = 0
            peakhigh_index = 0  
            information = 0
            accumulator1 = accumulator2 = accumulator3 = accumulator4 = accumulator5 = accumulator6 = 0
            pass
        if format_bits & 0x2:
            accumulator7 = evt_data_32[offset+0]
            accumulator8 = evt_data_32[offset+1]
            offset += 2
        else:
            accumulator7 = accumulator8 = 0
            pass
        if format_bits & 0x4:
            mawMax = evt_data_32[offset+0]
            maw_before = evt_data_32[offset+1]
            maw_after = evt_data_32[offset+2]
            offset += 3
        else:
            mawMax = maw_before = maw_after = 0
            pass
        if format_bits & 0x8:
            energy_first = evt_data_32[offset+0]
            energy = evt_data_32[offset+1]
            offset += 2
        else:
            energy_first = energy = 0
            pass
        wf_length_32 = (evt_data_32[offset+0]) & 0x03ffffff
        offset += 1 #now the offset points to the wf data
        fadcID = fadcIndex
        channel = channelIndex
        
        
        # compute expected and actual array dimensions
        wf_length16 = 2 * wf_length_32
        header_length16 = offset * 2
        expected_wf_length = len(evt_data_16) - header_length16

        # error check: waveform size must match expectations
        if wf_length16 != expected_wf_length:
            print(len(evt_data_16), header_length16)
            print("ERROR: Waveform size %d doesn't match expected size %d." %
                  (wf_length16, expected_wf_length))
            exit()

        # indexes of stuff (all referring to the 16 bit array)
        i_wf_start = header_length16
        i_wf_stop = i_wf_start + wf_length16

        # handle the waveform(s)
        if wf_length_32 > 0:
            wf_data = evt_data_16[i_wf_start:i_wf_stop]

        if len(wf_data) != expected_wf_length:
            print("ERROR: event %d, we expected %d WF samples and only got %d" %
                  (ievt, expected_wf_length, len(wf_data)))
            exit()

        # final raw wf array
        waveform = wf_data

        # if the wf is too big for pytables, we can window it
        if self.window:
            wf = Waveform(wf_data, self.sample_period, self.decoder_name)
            win_wf, win_ts = wf.window_waveform(self.win_type,
                                                self.n_samp,
                                                self.n_blsamp,
                                                test=False)
            # ts_lo, ts_hi = win_ts[0], win_ts[-1]  # FIXME: what does this mean?

            waveform = win_wf # modify final wf array

            if wf.is_garbage:
                ievt = self.ievt_gbg
                self.ievt_gbg += 1
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


class CAENDT57XX(DataTaker):
    """
    decode CAENDT5725 or CAENDT5730 digitizer data.
    
    Setting the model_name will set the appropriate sample_rate
    Use the input_config function to set certain variables by passing
    a dictionary, this will most importantly assemble the file header used
    by CAEN CoMPASS to label output files.
    """
    def __init__(self, *args, **kwargs):
        self.id = None
        self.model_name = "DT5725" # hack -- can't set the model name in the init
        self.decoder_name = "caen"
        self.file_header = None
        self.adc_bitcount = 14
        self.sample_rates = {"DT5725": 250e6, "DT5730": 500e6}
        self.sample_rate = None
        if self.model_name in self.sample_rates.keys():
            self.sample_rate = self.sample_rates[self.model_name]
        else:
            raise TypeError("Unidentified digitizer type: "+str(model_name))
        self.v_range = 2.0

        self.e_cal = None
        self.e_type = None
        self.int_window = None
        self.parameters = ["TIMETAG", "ENERGY", "E_SHORT", "FLAGS"]

        self.decoded_values = {
            "board": None,
            "channel": None,
            "timestamp": None,
            "energy": None,
            "energy_short": None,
            "flags": None,
            "num_samples": None,
            "waveform": []
        }
        super().__init__(*args, **kwargs)


    def input_config(self, config):
        self.id = config["id"]
        self.v_range = config["v_range"]
        self.e_cal = config["e_cal"]
        self.e_type = config["e_type"]
        self.int_window = config["int_window"]
        self.file_header = "CH_"+str(config["channel"])+"@"+self.model_name+"_"+str(config["id"])+"_Data_"


    def get_event_size(self, t0_file):
        with open(t0_file, "rb") as file:
            if self.e_type == "uncalibrated":
                first_event = file.read(24)
                [num_samples] = np.frombuffer(first_event[20:24], dtype=np.uint16)
                return 24 + 2*num_samples
            elif self.e_type == "calibrated":
                first_event = file.read(30)
                [num_samples] = np.frombuffer(first_event[26:30], dtype=np.uint32)
                return 30 + 2 * num_samples  # number of bytes / 2
            else:
                raise TypeError("Invalid e_type! Valid e_type's: uncalibrated, calibrated")


    def get_event(self, event_data_bytes):
        self.decoded_values["board"] = np.frombuffer(event_data_bytes[0:2], dtype=np.uint16)[0]
        self.decoded_values["channel"] = np.frombuffer(event_data_bytes[2:4], dtype=np.uint16)[0]
        self.decoded_values["timestamp"] = np.frombuffer(event_data_bytes[4:12], dtype=np.uint64)[0]
        if self.e_type == "uncalibrated":
            self.decoded_values["energy"] = np.frombuffer(event_data_bytes[12:14], dtype=np.uint16)[0]
            self.decoded_values["energy_short"] = np.frombuffer(event_data_bytes[14:16], dtype=np.uint16)[0]
            self.decoded_values["flags"] = np.frombuffer(event_data_bytes[16:20], np.uint32)[0]
            self.decoded_values["num_samples"] = np.frombuffer(event_data_bytes[20:24], dtype=np.uint32)[0]
            self.decoded_values["waveform"] = np.frombuffer(event_data_bytes[24:], dtype=np.uint16)
        elif self.e_type == "calibrated":
            self.decoded_values["energy"] = np.frombuffer(event_data_bytes[12:20], dtype=np.float64)[0]
            self.decoded_values["energy_short"] = np.frombuffer(event_data_bytes[20:22], dtype=np.uint16)[0]
            self.decoded_values["flags"] = np.frombuffer(event_data_bytes[22:26], np.uint32)[0]
            self.decoded_values["num_samples"] = np.frombuffer(event_data_bytes[26:30], dtype=np.uint32)[0]
            self.decoded_values["waveform"] = np.frombuffer(event_data_bytes[30:], dtype=np.uint16)
        else:
            raise TypeError("Invalid e_type! Valid e_type's: uncalibrated, calibrated")
        return self.assemble_data_row()


    def assemble_data_row(self):
        timestamp = self.decoded_values["timestamp"]
        energy = self.decoded_values["energy"]
        energy_short = self.decoded_values["energy_short"]
        flags = self.decoded_values["flags"]
        waveform = self.decoded_values["waveform"]
        return [timestamp, energy, energy_short, flags], waveform


    def create_dataframe(self, array):
        waveform_labels = [str(item) for item in list(range(self.decoded_values["num_samples"]-1))]
        column_labels = self.parameters + waveform_labels
        dataframe = pd.DataFrame(data=array, columns=column_labels, dtype=float)
        return dataframe


class ORCAGretina4M(DataTaker):
    """ 
    decode Majorana Gretina4M digitizer data
    
    NOTE: Tom Caldwell made some nice new summary slides on a 2019 LEGEND call
    https://indico.legend-exp.org/event/117/contributions/683/attachments/467/717/mjd_data_format.pdf
    """
    def __init__(self, *args, **kwargs):
        self.decoder_name = 'ORGretina4MWaveformDecoder'
        self.class_name = 'ORGretina4MModel'
        self.decoded_values = {
            "packet_id": [],
            "ievt": [],
            "energy": [],
            "timestamp": [],
            "channel": [],
            "board_id": [],
            "waveform": [],
        }
        super().__init__(*args, **kwargs)
        self.chan_list = None
        self.is_multisampled = True
        self.event_header_length = 18
        self.sample_period = 10  # ns
        self.gretina_event_no = 0
        self.window = False
        self.n_blsamp = 500
        self.ievt = 0
        
        self.df_metadata = None # hack, this probably isn't right
        self.active_channels = self.find_active_channels()
        

    def crate_card_chan(self, crate, card, channel):
        return (crate << 9) + (card << 4) + (channel)


    def find_active_channels(self):
        """ 
        Only do this for multi-detector data 
        """
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
        """ 
        Parse the header for an individual event 
        """
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

        # if the wf is too big for pytables, we can window it
        if self.window:
            wf = Waveform(wf_data, self.sample_period, self.decoder_name)
            waveform = wf.window_waveform(self.win_type,
                                          self.n_samp,
                                          self.n_blsamp,
                                          test=False)
            if wf.is_garbage:
                ievt = self.ievt_gbg
                self.ievt_gbg += 1
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


class SIS3316ORCADecoder(DataTaker):
    """ 
    handle ORCA Struck 3316 digitizer 
    
    TODO: 
    handle per-channel data (gain, ...)
    most metadata of Struck header (energy, ...)
    """
    def __init__(self, *args, **kwargs):
        
        self.decoder_name = 'ORSIS3316WaveformDecoder'
        self.class_name = 'ORSIS3316Model'

        # store an entry for every event
        self.decoded_values = {
            "packet_id": [],
            "ievt": [],
            "energy_first": [],
            "energy": [],
            "timestamp": [],
            "channel": [],
            "waveform": [],
        }
        super().__init__(*args, **kwargs) # also initializes the garbage df (whatever that means...)

        # self.event_header_length = 1 #?
        self.sample_period = 10  # ns, I will set this later, according to header info
        self.gain = 0           
        self.h5_format = "table"
        self.ievt = 0       #event number
        self.ievt_gbg = 0      #garbage event number
        self.window = False
        
        
    def decode_event(self, event_data_bytes, packet_id, header_dict, 
                     verbose=False):

        # parse the raw event data into numpy arrays of 16 and 32 bit ints
        evt_data_32 = np.fromstring(event_data_bytes, dtype=np.uint32)
        evt_data_16 = np.fromstring(event_data_bytes, dtype=np.uint16)

        #TODO Figure out the header, particularly card/crate/channel/timestamp
        n_lost_msb = 0
        n_lost_lsb = 0
        n_lost_records = 0
        crate = evt_data_32[3]
        card = evt_data_32[4]
        channel = evt_data_32[4]
        buffer_wrap = 0
        crate_card_chan = crate + card + channel
        wf_length_32 = 0
        ene_wf_length = evt_data_32[4]
        evt_header_id = 0
        timestamp = 0

        # compute expected and actual array dimensions
        wf_length16 = 1024
        orca_helper_length16 = 52
        header_length16 = orca_helper_length16
        ene_wf_length16 = 2 * ene_wf_length
        footer_length16 = 0

        expected_wf_length = (len(evt_data_16) - header_length16 - ene_wf_length16)/2

        if wf_length16 != expected_wf_length:
            print("ERROR: Waveform size %d doesn't match expected size %d." %
                  (wf_length16, expected_wf_length))
            #exit()

        # indexes of stuff (all referring to the 16 bit array)
        i_wf_start = header_length16
        i_wf_stop = i_wf_start + wf_length16
        i_ene_start = i_wf_stop + 1
        i_ene_stop = i_ene_start + ene_wf_length16


        # handle the waveform(s)
        if wf_length16 > 0:
            wf_data = evt_data_16[i_wf_start:i_wf_stop]


        #TODO check if number of events matches expected
        #if len(wf_data) != expected_wf_length:
        #    print("ERROR: We expected %d WF samples and only got %d" %
        #          (expected_wf_length, len(wf_data)))
        #    exit()

        # final raw wf array
        waveform = wf_data

        # set the event number (searchable HDF5 column)
        ievt = self.ievt
        self.ievt += 1

        # send any variable with a name in "decoded_values" to the pandas output
        self.format_data(locals())


class FlashCam(DataTaker):
    """ 
    decode FlashCam digitizer data.
    """
    def __init__(self, *args, **kwargs):
        """
        """
        self.decoder_name = "FlashCam"
        
        # these are read for every event (decode_event)
        self.decoded_values = {
          "ievt": [], # index of event
          "timestamp": [], # time since beginning of file
          "channel": [], # right now, index of the trigger (trace)
          "baseline" : [], # averages prebaseline0 and prebaseline1
          "wf_max": [], # ultra-simple np.max energy estimation
          "wf_std": [], # ultra-simple np.std noise estimation
          "waveform": [] # digitizer data
        }
        
        # these are read for every file (get_file_config)
        self.config_names = [
            "nsamples", # samples per channel
            "nadcs", # number of adc channels
            "ntriggers", # number of triggertraces
            "telid", # id of telescope
            "adcbits", # bit range of the adc channels
            "sumlength", # length of the fpga integrator
            "blprecision", # precision of the fpga baseline
            "mastercards", # number of attached mastercards
            "triggercards", # number of attached triggercards
            "adccards", # number of attached fadccards
            "gps", # gps mode (0: not used, 1: external pps and 10MHz)
            ]
        
        # put add'l info useful for LH5 specification
        # default structure is array<1>{real}, default unit is None.
        # here we only specify columns if they are non-default.
        self.lh5_spec = {
            "timestamp":{"units":"sec"},
            "baseline":{"units":"adc"},
            "wf_max":{"units":"adc"},
            "wf_std":{"units":"adc"},
        }
        
        super().__init__(*args, **kwargs)
        
        
    def get_file_config(self, fcio):
        """
        access FCIOConfig members once when each file is opened
        """
        self.file_config = {c:getattr(fcio, c) for c in self.config_names}

          
    def decode_event(self, fcio, packet_id, verbose=False):
        """
        access FCIOEvent members for each event in the raw file
        """
        ievt = fcio.eventnumber # the eventnumber since the beginning of the file
        timestamp = fcio.eventtime  # the time since the beginning of the file in seconds
        traces = fcio.traces # the full traces for the event: (nadcs, nsamples)
        baselines = fcio.baselines # the fpga baseline values for each channel in LSB
        # baselines = fcio.average_prebaselines # equivalent?
        
        # these are empty in my test file
        integrals = fcio.integrals # the fpga integrator values for each channel in LSB
        triggertraces = fcio.triggertraces # the triggersum traces: (ntriggers, nsamples)
        
        # all channels are read out simultaneously for each event
        for iwf in range(self.file_config["nadcs"]):
            channel = iwf
            waveform = traces[iwf]
            baseline = baselines[iwf]
            wf_max = np.amax(waveform)
            wf_std = np.std(waveform)
            self.total_count += 1
            
            # i don't know what indicates a garbage event yet
            # if wf.is_garbage:
            #     self.garbage_count += 1
            #     self.format_data(locals(), wf.is_garbage)
            #     return
            
            # send any variable with a name in "decoded_values" to the output
            self.format_data(locals())  


