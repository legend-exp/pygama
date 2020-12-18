import sys
import numpy as np

from .orcadaq import OrcaDecoder

class ORCAStruck3302(OrcaDecoder):
    """
    decode ORCA Struck 3302 digitizer data
    """
    def __init__(self, *args, **kwargs):

        self.decoder_name = 'ORSIS3302DecoderForEnergy'
        self.orca_class_name = 'ORSIS3302Model'

        self.decoded_values = {
            'packet_id': {
               'dtype': 'uint32',
             },
            'ievt': {
              'dtype': 'uint32',
            },
            'energy': {
              'dtype': 'uint32',
              'units': 'adc',
            },
            'energy_first': {
              'dtype': 'uint32',
            },
            'timestamp': {
              'dtype': 'uint32',
              'units': 'clock_ticks',
            },
            'crate': {
              'dtype': 'uint8',
            },
            'card': {
              'dtype': 'uint8',
            },
            'channel': {
              'dtype': 'uint8',
            },
            'waveform': {
              'dtype': 'uint16',
              'datatype': 'waveform',
              'length': 65532, # max value. override this before initalizing buffers to save RAM
              'sample_period': 10, # override if a different clock rate is used
              'sample_period_units': 'ns',
              'units': 'adc',
            },
        }
        super().__init__(*args, **kwargs) # also initializes the garbage df
        self.enabled_cccs = []
        self.ievt = 0


    def set_object_info(self, object_info):
        self.object_info = object_info

        # parse object_info for important info
        for card_dict in self.object_info:
            crate = card_dict['Crate']
            card = card_dict['Card']

            int_enabled_mask = card_dict['internalTriggerEnabledMask']
            ext_enabled_mask = card_dict['externalTriggerEnabledMask']
            enabled_mask = int_enabled_mask | ext_enabled_mask
            trace_length = 0
            for channel in range(8):
                # only care about enabled channels
                if (enabled_mask >> channel) & 0x1:
                    # save list of enabled channels
                    #self.enabled_cccs.append(get_ccc(crate, card, channel))

                    # get trace length(s). Should all be the same until
                    # multi-buffer mode is implemented AND each channel has its
                    # own buffer
                    this_length = card_dict['sampleLengths'][int(channel/2)]
                    if trace_length == 0: trace_length = this_length
                    elif this_length != trace_length:
                        print('SIS3316ORCADecoder Error: multiple trace lengths not supported')
                        sys.exit()

            # check trace length and update decoded_values
            if trace_length <= 0 or trace_length > 2**16:
                print('SIS3316ORCADecoder Error: invalid trace_length', trace_length)
                sys.exit()
            self.decoded_values['waveform']['length'] = trace_length


    def decode_packet(self, packet, lh5_table, packet_id, header_dict, verbose=False):
        """
        see README for the 32-bit data word diagram
        """

        # interpret the raw event data into numpy arrays of 16 and 32 bit ints
        # does not copy data. p32 and p16 are read-only
        p32 = np.frombuffer(packet, dtype=np.uint32)
        p16 = np.frombuffer(packet, dtype=np.uint16)

        # aliases for brevity
        tb = lh5_table
        ii = tb.loc

        # store packet id
        tb['packet_id'].nda[ii] = packet_id

        # start reading the binary, baby
        n_lost_msb = (p32[0] >> 25) & 0x7F
        n_lost_lsb = (p32[0] >> 2) & 0x7F
        n_lost_records = (n_lost_msb << 7) + n_lost_lsb
        tb['crate'].nda[ii] = (p32[0] >> 21) & 0xF
        tb['card'].nda[ii] = (p32[0] >> 16) & 0x1F
        tb['channel'].nda[ii] = (p32[0] >> 8) & 0xFF
        buffer_wrap = p32[0] & 0x1
        #crate_card_chan = (tb['crate'].nda[ii] << 9) + (tb['card'].nda[ii] << 4) + tb['channel'].nda[ii]
        wf_length32 = p32[1]
        ene_wf_length32 = p32[2]
        evt_header_id = p32[3] & 0xFF
        tb['timestamp'].nda[ii] = p32[4] + ((p32[3] & 0xFFFF0000) << 16)
        last_word = p32[-1]

        # get the footer
        tb['energy'].nda[ii] = p32[-4]
        tb['energy_first'].nda[ii] = p32[-3]
        extra_flags = p32[-2]

        # compute expected and actual array dimensions
        wf_length16 = 2 * wf_length32
        orca_helper_length16 = 2
        sis_header_length16 = 12 if buffer_wrap else 8
        header_length16 = orca_helper_length16 + sis_header_length16
        ene_wf_length16 = 2 * ene_wf_length32
        footer_length16 = 8
        expected_wf_length = len(p16) - orca_helper_length16 - sis_header_length16 - \
            footer_length16 - ene_wf_length16

        # error check: waveform size must match expectations
        if wf_length16 != expected_wf_length or last_word != 0xdeadbeef:
            print(len(p16), orca_helper_length16, sis_header_length16,
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
            i_start_1 = p32[6] + header_length16 + 1
            i_stop_1 = i_wf_stop  # end of the wf record
            i_start_2 = i_wf_start  # beginning of the wf record
            i_stop_2 = i_start_1

        # handle the waveform(s)
        #energy_wf = np.zeros(ene_wf_length16)  # not used rn
        tbwf = tb['waveform']['values'].nda[ii]
        if wf_length32 > 0:
            if not buffer_wrap:
                if i_wf_stop - i_wf_start != expected_wf_length:
                    print("ERROR: event %d, we expected %d WF samples and only got %d" %
                          (ievt, expected_wf_length, i_wf_stope - i_wf_start))
                tbwf[:expected_wf_length] = p16[i_wf_start:i_wf_stop]
            else:
                len1 = i_stop_1-i_start_1
                len2 = i_stop_2-i_start_2
                if len1+len2 != expected_wf_length:
                    print("ERROR: event %d, we expected %d WF samples and only got %d" %
                          (ievt, expected_wf_length, len1+len2))
                    exit()
                tbwf[:len1] = p16[i_start_1:i_stop_1]
                tbwf[len1:len1+len2] = p16[i_start_2:i_stop_2]


        # set the event number (searchable HDF5 column)
        tb['ievt'].nda[ii] = self.ievt
        self.ievt += 1
        tb.push_row()


'''
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
'''
