import copy
import logging
import sys

import numpy as np

from .orca_base import OrcaDecoder, get_ccc

log = logging.getLogger(__name__)

class ORSIS3302DecoderForEnergy(OrcaDecoder):
    """
    Decoder for Struck 3302 digitizer data written by ORCA
    """
    def __init__(self, header=None, **kwargs):
        """
        DOCME
        """
        self.decoded_values_template = {
            'packet_id': { 'dtype': 'uint32', },
            'energy': { 'dtype': 'uint32', 'units': 'adc', },
            'energy_first': { 'dtype': 'uint32', },
            'timestamp': { 'dtype': 'uint64', 'units': 'clock_ticks', },
            'crate': { 'dtype': 'uint8', },
            'card': { 'dtype': 'uint8', },
            'channel': { 'dtype': 'uint8', },
            'waveform': {
                'dtype': 'uint16',
                'datatype': 'waveform',
                'wf_len': 65532, # max value. override this before initializing buffers to save RAM
                'dt': 10, # override if a different clock rate is used
                'dt_units': 'ns',
                't0_units': 'ns', },
        }
        self.decoded_values = {}
        super().__init__(header=header, **kwargs)
        self.skipped_channels = {}

    def set_header(self, header):
        self.header = header

        obj_info_dict = header.get_object_info('ORSIS3302Model')
        # Loop over crates, cards, build decoded values for enabled channels
        for crate in obj_info_dict:
            for card in obj_info_dict[crate]:
                int_enabled_mask = obj_info_dict[crate][card]['internalTriggerEnabledMask']
                ext_enabled_mask = obj_info_dict[crate][card]['externalTriggerEnabledMask']
                enabled_mask = int_enabled_mask | ext_enabled_mask
                for channel in range(8):
                    # only care about enabled channels
                    if not ((enabled_mask >> channel) & 0x1): continue

                    ccc = get_ccc(crate, card, channel)

                    self.decoded_values[ccc] = copy.deepcopy(self.decoded_values_template)

                    # get trace length(s). Should all be the same until
                    # multi-buffer mode is implemented AND each channel has its
                    # own buffer
                    trace_length = obj_info_dict[crate][card]['sampleLengths'][int(channel/2)]
                    if trace_length <= 0 or trace_length > 2**16:
                        print('SIS3316ORCADecoder Error: invalid trace_length', trace_length)
                        sys.exit()
                    self.decoded_values[ccc]['waveform']['wf_len'] = trace_length

    def get_key_list(self):
        key_list = []
        for key in self.decoded_values.keys():
            key_list += [key]
        return key_list

    def get_decoded_values(self, key=None):
        if key is None:
            dec_vals_list = self.decoded_values.values()
            if len(dec_vals_list) >= 0: return list(dec_vals_list)[0]
            raise RuntimeError('decoded_values not built')
        if key in self.decoded_values: return self.decoded_values[key]
        raise KeyError(f'no decoded values for key {key}')

    def decode_packet(self, packet, packet_id, rbl):
        ''' decode the orca struck 3302 packet '''
        evt_rbkd = rbl.get_keyed_dict()

        # read the crate/card/channel first
        crate              = (packet[1] >> 21) & 0xF
        card               = (packet[1] >> 16) & 0x1F
        channel            = (packet[1] >> 8) & 0xFF
        ccc = get_ccc(crate, card, channel)

        # get the table for this crate/card/channel
        if ccc not in evt_rbkd:
            if ccc not in self.skipped_channels:
                self.skipped_channels[ccc] = 0
                print(f'Skipping channel: {ccc}')
                print(f'evt_rbkd: {evt_rbkd.keys()}')
            self.skipped_channels[ccc] += 1
            return False
        tbl = evt_rbkd[ccc].lgdo
        ii = evt_rbkd[ccc].loc

        # store packet id
        tbl['packet_id'].nda[ii] = packet_id

        # read the rest of the record
        # n_lost_msb = (packet[1] >> 25) & 0x7F
        # n_lost_lsb = (packet[1] >> 2) & 0x7F
        # n_lost_records = (n_lost_msb << 7) + n_lost_lsb
        tbl['crate'].nda[ii] = crate
        tbl['card'].nda[ii] = card
        tbl['channel'].nda[ii] = channel
        buffer_wrap = packet[1] & 0x1
        wf_length32 = packet[2]
        ene_wf_length32 = packet[3]
        # evt_header_id = packet[4] & 0xFF
        tbl['timestamp'].nda[ii] = packet[5] + ((packet[4] & 0xFFFF0000) << 16) # might need to convert to uint64
        last_word = packet[-1]

        # get the footer
        tbl['energy'].nda[ii] = packet[-4]
        tbl['energy_first'].nda[ii] = packet[-3]
        # extra_flags = packet[-2]
        
        # interpret the raw event data into numpy array of 16 bit ints
        # does not copy data. p16 is read-only
        p16 = np.frombuffer(packet, dtype=np.uint16)

        # compute expected and actual array dimensions
        wf_length16 = 2 * wf_length32
        ene_wf_length16 = 2 * ene_wf_length32
        orca_header_length16 = 8
        sis_header_length16 = 8 if buffer_wrap else 4
        header_length16 = orca_header_length16 + sis_header_length16
        footer_length16 = 8
        expected_wf_length16 = len(p16) - header_length16 - footer_length16 - ene_wf_length16
            
        # error check: waveform size must match expectations       
        if wf_length16 != expected_wf_length16 or last_word != 0xdeadbeef:
            print(len(p16), orca_header_length16, sis_header_length16,
                  footer_length16, ene_wf_length16)
            print("ERROR: Waveform size %d doesn't match expected size %d." %
                  (wf_length16, expected_wf_length16))
            print("       The Last Word (should be 0xdeadbeef):",
                  hex(last_word))
            exit()

        # splitting waveform indices into two chunks (all referring to the 16 bit array) 
        i_wf_start = header_length16
        i_wf_stop = i_wf_start + wf_length16
        # i_ene_start = i_wf_stop + 1
        # i_ene_stop = i_ene_start + ene_wf_length16
        if buffer_wrap:
            # start somehwere in the middle of the record
            i_start_1 = packet[7] + header_length16 + 1
            i_stop_1 = i_wf_stop  # end of the wf record
            i_start_2 = i_wf_start  # beginning of the wf record
            i_stop_2 = i_start_1

        # handle the waveform(s)
        tbwf = tbl['waveform']['values'].nda[ii]
        if wf_length32 > 0:
            if not buffer_wrap:
                if i_wf_stop - i_wf_start != expected_wf_length16:
                    print("ERROR: event %d, we expected %d WF samples and only got %d" %
                        (expected_wf_length16, i_wf_stop - i_wf_start))
                tbwf[:expected_wf_length16] = p16[i_wf_start:i_wf_stop]
            else:
                len1 = i_stop_1-i_start_1
                len2 = i_stop_2-i_start_2
                if len1+len2 != expected_wf_length16:
                    print("ERROR: event %d, we expected %d WF samples and only got %d" %
                          (expected_wf_length16, len1+len2))
                    exit()
                tbwf[:len1] = p16[i_start_1:i_stop_1]
                tbwf[len1:len1+len2] = p16[i_start_2:i_stop_2]             
                
        evt_rbkd[ccc].loc += 1
        return evt_rbkd[ccc].is_full()
