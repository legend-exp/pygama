import copy

import numpy as np

from ..fc.fc_event_decoder import fc_decoded_values
from .orca_base import OrcaDecoder, get_ccc


class ORCAFlashCamListenerConfigDecoder(OrcaDecoder):
    """
    '''
    Decoder for FlashCam listener config written by ORCA
    '''
    def __init__(self, *args, **kwargs):

        self.decoder_name    = 'ORFlashCamListenerConfigDecoder'
        self.orca_class_name = 'ORFlashCamListenerModel'

        # up through ch_inputnum, these are in order of the fcio data format
        # for similicity.  append any additional values after this.
        self.decoded_values = {
            'readout_id':   { 'dtype': 'uint16', },
            'listener_id':  { 'dtype': 'uint16', },
            'telid':        { 'dtype': 'int32',  },
            'nadcs':        { 'dtype': 'int32',  },
            'ntriggers':    { 'dtype': 'int32',  },
            'nsamples':     { 'dtype': 'int32',  },
            'adcbits':      { 'dtype': 'int32',  },
            'sumlength':    { 'dtype': 'int32',  },
            'blprecision':  { 'dtype': 'int32',  },
            'mastercards':  { 'dtype': 'int32',  },
            'triggercards': { 'dtype': 'int32',  },
            'adccards':     { 'dtype': 'int32',  },
            'gps':          { 'dtype': 'int32',  },
            'ch_boardid':   { 'dtype': 'uint16',
                              'datatype':
                              'array_of_equalsized_arrays<1,1>{real}',
                              'length': 2400, },
            'ch_inputnum':  { 'dtype': 'uint16',
                              'datatype':
                              'array_of_equalsized_arrays<1,1>{real}',
                              'length': 2400, },
            }

        super().__init__(args, kwargs)


    def get_decoded_values(self, channel=None):
        return self.decoded_values


    def max_n_rows_per_packet(self):
        return 1


    def decode_packet(self, packet, lh5_tables,
                      packet_id, header_dict, verbose=False):

        data = np.frombuffer(packet, dtype=np.int32)
        tbl  = lh5_tables
        ii   = tbl.loc

        tbl['readout_id'].nda[ii]  = (data[0] & 0xffff0000) >> 16
        tbl['listener_id'].nda[ii] =  data[0] & 0x0000ffff

        for i,k in enumerate(self.decoded_values):
            if i < 2: continue
            tbl[k].nda[ii] = data[i-1]
            if k == 'gps': break

        data = np.frombuffer(packet, dtype=np.uint32)
        data = data[list(self.decoded_values.keys()).index('ch_boardid')-1:]
        for i in range(len(data)):
            tbl['ch_boardid'].nda[ii][i]  = (data[i] & 0xffff0000) >> 16
            tbl['ch_inputnum'].nda[ii][i] =  data[i] & 0x0000ffff

        tbl.push_row()


class ORCAFlashCamListenerStatusDecoder(OrcaDecoder):
    '''
    Decoder for FlashCam status packets written by ORCA

    Some of the card level status data contains an  array of values
    (temperatures for instance) for each card.  Since lh5 currently only
    supports a 1d vector of 1d vectors, this (card,value) data has to be
    flattened before populating the lh5 table.
    '''
    def __init__(self, *args, **kwargs):

        self.decoder_name    = 'ORFlashCamListenerStatusDecoder'
        self.orca_class_name = 'ORFlashCamListenerModel'
        self.nOtherErrors    = np.uint32(5)
        self.nEnvMonitors    = np.uint32(16)
        self.nCardTemps      = np.uint32(5)
        self.nCardVoltages   = np.uint32(6)
        self.nADCTemps       = np.uint32(2)
        self.nCTILinks       = np.uint32(4)
        self.nCards          = np.uint32(1)

        self.decoded_values = {
            'readout_id':  { 'dtype': 'uint16', },
            'listener_id': { 'dtype': 'uint16', },
            'cards':       { 'dtype': 'int32',  },
            'status':      { 'dtype': 'int32',  },
            'statustime':  { 'dtype': 'float64', 'units': 's', },
            'cputime':     { 'dtype': 'float64', 'units': 's', },
            'startoffset': { 'dtype': 'float64', 'units': 's', },
            'card_fcio_id':  {
                'dtype':        'uint32',
                'datatype':     'array<1>{array<1>{real}}',
                'length_guess':  self.nCards, },
            'card_status': {
                'dtype':        'uint32',
                'datatype':     'array<1>{array<1>{real}}',
                'length_guess':  self.nCards, },
            'card_event_number': {
                'dtype':        'uint32',
                'datatype':     'array<1>{array<1>{real}}',
                'length_guess':  self.nCards, },
            'card_pps_count': {
                'dtype':        'uint32',
                'datatype':     'array<1>{array<1>{real}}',
                'length_guess':  self.nCards, },
            'card_tick_count': {
                'dtype':        'uint32',
                'datatype':     'array<1>{array<1>{real}}',
                'length_guess':  self.nCards, },
            'card_max_ticks': {
                'dtype':        'uint32',
                'datatype':     'array<1>{array<1>{real}}',
                'length_guess':  self.nCards, },
            'card_total_errors': {
                'dtype':        'uint32',
                'datatype':     'array<1>{array<1>{real}}',
                'length_guess':  self.nCards, },
            'card_env_errors': {
                'dtype':        'uint32',
                'datatype':     'array<1>{array<1>{real}}',
                'length_guess':  self.nCards, },
            'card_cti_errors': {
                'dtype':        'uint32',
                'datatype':     'array<1>{array<1>{real}}',
                'length_guess':  self.nCards, },
            'card_link_errors': {
                'dtype':        'uint32',
                'datatype':     'array<1>{array<1>{real}}',
                'length_guess':  self.nCards, },
            'card_other_errors': {
                'dtype':        'uint32',
                'datatype':     'array<1>{array<1>{real}}',
                'length_guess':  self.nCards * self.nOtherErrors, },
            'card_temp': {
                'dtype':        'uint32',
                'datatype':     'array<1>{array<1>{real}}',
                'length_guess':  self.nCards * self.nCardTemps,
                'units':        'mC', },
            'card_voltage': {
                'dtype':        'uint32',
                'datatype':     'array<1>{array<1>{real}}',
                'length_guess':  self.nCards * self.nCardVoltages,
                'units':        'mV', },
            'card_current': {
                'dtype':        'uint32',
                'datatype':     'array<1>{array<1>{real}}',
                'length_guess':  self.nCards,
                'units':        'mA', },
            'card_humidity': {
                'dtype':        'uint32',
                'datatype':     'array<1>{array<1>{real}}',
                'length_guess':  self.nCards,
                'units':        'o/oo', },
            'card_adc_temp': {
                'dtype':        'uint32',
                'datatype':     'array<1>{array<1>{real}}',
                'length_guess':  self.nCards * self.nADCTemps,
                'units':        'mC', },
            'card_cti_link': {
                'dtype':        'uint32',
                'datatype':     'array<1>{array<1>{real}}',
                'length_guess':  self.nCards * self.nCTILinks, },
            'card_card_link_state': {
                'dtype':        'uint32',
                'datatype':     'array<1>{array<1>{real}}',
                'length_guess':  self.nCards * self.nCards, },
        }

        # arrays to temporarily store card-level decoded data
        self.cdata = {}
        self.resize_card_data(ncards=self.nCards)

        super().__init__(args, kwargs)


    def resize_card_data(self, ncards):
        try: ncards = np.uint32(ncards)
        except ValueError: return
        if ncards == 0: return
        for key in self.decoded_values:
            # ignore keys that aren't card level
            if key.find('card_') != 0: continue
            try:
                # skip keys for things that aren't arrays with a length_guess
                if self.decoded_values[key]['datatype'].find('array') != 0:
                    continue
                length = self.decoded_values[key]['length_guess']
                try:
                    # resize if ncards differs from the old shape
                    oldshape = self.cdata[key].shape
                    if oldshape[0] == ncards: continue
                    if key.find('card_card_') == 0:
                        self.cdata[key].resize((ncards,ncards,) + oldshape[2:])
                    else:
                        self.cdata[key].resize((ncards,) + oldshape[1:])
                except KeyError:
                    # if the key didn't exist set the ndarray for this key
                    if ((length == ncards or (length % ncards) != 0) and
                        key.find('card_card_') == -1):
                        self.cdata[key] = np.ndarray(shape=(length),
                                                     dtype=np.uint32)
                    else:
                        nval = np.uint32(length / ncards)
                        self.cdata[key] = np.ndarray(shape=(ncards, nval),
                                                     dtype=np.uint32)
            except KeyError: continue
        # set nCards to allow for not calling this function during decoding
        self.nCards = ncards


    def get_decoded_values(self, channel=None):
        return self.decoded_values


    def max_n_rows_per_packet(self):
        return 1


    def decode_packet(self, packet, lh5_tables,
                      packet_id, header_dict, verbose=False):

        data = np.frombuffer(packet, dtype=np.uint32)
        tbl  = lh5_tables
        ii   = tbl.loc

        # populate the packet header information
        tbl['readout_id'].nda[ii]  = (data[0] & 0xffff0000) >> 16
        tbl['listener_id'].nda[ii] =  data[0] & 0x0000ffff
        tbl['status'].nda[ii]      = np.int32(data[1])
        tbl['statustime'].nda[ii]  = np.float64(data[2] + data[3] / 1.0e6)
        tbl['cputime'].nda[ii]     = np.float64(data[4] + data[5] / 1.0e6)
        tbl['startoffset'].nda[ii] = np.float64(data[7] + data[8] / 1.0e6)
        tbl['cards'].nda[ii]       = np.int32(data[12])

        # resize the card level data if necessary
        if data[12] != self.nCards:
            print('ORlashCamListenerStatusDecoder: resizing card arrays '
                  'from', self.nCards, ' cards to', data[12])
            self.resize_card_data(ncards=data[12])

        # set the local card level data
        for i in range(np.int(data[12])):
            j = 14 + i * (data[12] + 14 + self.nOtherErrors +
                          self.nEnvMonitors + self.nCTILinks)

            self.cdata['card_fcio_id'][i]      = data[j]
            self.cdata['card_status'][i]       = data[j+1]
            self.cdata['card_event_number'][i] = data[j+2]
            self.cdata['card_pps_count'][i]    = data[j+3]
            self.cdata['card_tick_count'][i]   = data[j+4]
            self.cdata['card_max_ticks'][i]    = data[j+5]
            self.cdata['card_total_errors'][i] = data[j+10]
            self.cdata['card_env_errors'][i]   = data[j+11]
            self.cdata['card_cti_errors'][i]   = data[j+12]
            self.cdata['card_link_errors'][i]  = data[j+13]
            k = j + 14
            self.cdata['card_other_errors'][i][:]= data[k:k+self.nOtherErrors]
            k += self.nOtherErrors
            self.cdata['card_temp'][i][:]        = data[k:k+self.nCardTemps]
            k += self.nCardTemps
            self.cdata['card_voltage'][i][:]     = data[k:k+self.nCardVoltages]
            k += self.nCardVoltages
            self.cdata['card_current'][i]        = data[k]
            self.cdata['card_humidity'][i]       = data[k+1]
            k += 2
            self.cdata['card_adc_temp'][i][:]    = data[k:k+self.nADCTemps]
            k += self.nADCTemps
            self.cdata['card_cti_link'][i][:]    = data[k:k+self.nCTILinks]
            k += self.nCTILinks
            self.cdata['card_card_link_state'][i][:]  = data[k:k+data[12]]

        # populate the card level data with the flattened local data, then push
        for key in self.cdata:
            tbl[key].set_vector(ii, self.cdata[key].flatten())

        tbl.push_row()
    """

class ORFlashCamADCWaveformDecoder(OrcaDecoder):
    """
    Decoder for FlashCam ADC data written by ORCA
    """
    def __init__(self, *args, **kwargs):
        """
        DOCME
        """
        # start with the values defined in fcdaq
        self.decoded_values_template = copy.deepcopy(fc_decoded_values)
        # add header values from Orca
        self.decoded_values_template.update( {
            'crate' :   { 'dtype': 'uint8',  },
            'card' :    { 'dtype': 'uint8',  },
            'ch_orca' : { 'dtype': 'uint8',  },
            'fcio_id' : { 'dtype': 'uint16', }
        } )
        self.decoded_values = {}
        super().__init__(*args, **kwargs)
        self.skipped_channels = {}


    def set_header(self, header):
        self.header = header

        # get the readout list for looking up the waveform length.
        # catch AttributeError for when the header is not yet set.
        object_info = header.get_object_info('ORFlashCamADCModel')
        rol = []
        try: rol=header.get_readout_info('ORFlashCamListenerModel')
        except AttributeError: pass

        for card_dict in object_info:
            crate   = card_dict['Crate']
            card    = card_dict['Card']
            enabled = card_dict['Enabled']
            # find the listener id for this card from the readout list
            listener = -1
            for ro in rol:
                try:
                    for obj in ro['children']:
                        if obj['crate']   == crate and obj['station'] == card:
                            listener = ro['uniqueID']
                            break
                except KeyError: pass
            # with the listener id, find the event samples for that listener
            samples = 0
            if listener >= 0:
                aux = header.get_auxhw_info('ORFlashCamListenerModel', listener)
                for info in aux:
                    try: samples = max(samples, info['eventSamples'])
                    except KeyError: continue
            # for each enabled channel, set the decoded values and wf length
            for channel in range(len(enabled)):
                if not enabled[channel]: continue
                ccc = get_ccc(crate, card, channel)
                if samples > 0:
                    self.decoded_values[ccc] = copy.deepcopy(self.decoded_values_template)
                    self.decoded_values[ccc]['waveform']['wf_len'] = samples


    def get_key_list(self):
        return list(self.decoded_values.keys())


    def get_decoded_values(self, channel=None):
        if channel is None:
            dec_vals_list = self.decoded_values.items()
            if len(dec_vals_list) == 0:
                print('ORFlashCamADCWaveformDecoder: error - decoded_values not built')
                return None
            return list(dec_vals_list)[0][1] # return first thing found
        if channel in self.decoded_values: return self.decoded_values[channel]
        print('ORFlashCamADCWaveformDecoder: error - '
              'no decoded values for channel ', channel)
        return None


    def decode_packet(self, packet, packet_id, rbl, verbosity=0):
        evt_rbkd = rbl.get_keyed_dict()

        # unpack lengths and ids from the header words
        orca_header_length = (packet[1] & 0xf0000000) >> 28
        fcio_header_length = (packet[1] & 0x0fc00000) >> 22
        wf_samples         = (packet[1] & 0x003fffc0) >> 6
        crate              = (packet[2] & 0xf8000000) >> 27
        card               = (packet[2] & 0x07c00000) >> 22
        channel            = (packet[2] & 0x00003c00) >> 10
        fcio_id            =  packet[2] & 0x000003ff
        ccc = get_ccc(crate, card, channel)

        # get the table for this crate/card/channel
        if ccc not in evt_rbkd:
            if ccc not in self.skipped_channels:
                self.skipped_channels[ccc] = 0
            self.skipped_channels[ccc] += 1
            return False
        tbl = evt_rbkd[ccc].lgdo
        ii = evt_rbkd[ccc].loc

        # check that the waveform length is as expected
        rb_wf_len = tbl['waveform']['values'].nda.shape[1]
        if wf_samples != rb_wf_len:
            if not hasattr(self, 'wf_len_errs'): self.wf_len_errs = {}
            if ccc not in self.wf_len_errs:
                print(f'ORCAFlashCamADCWaveformDecoder warning: waveform from ccc {ccc} of length {wf_samples} with expected length {rb_wf_len}')
                self.wf_len_errs[ccc] = True
            if wf_samples > rb_wf_len: wf_samples = rb_wf_len

        # set the values decoded from the header words
        tbl['packet_id'].nda[ii] = packet_id
        tbl['crate'].nda[ii]     = crate
        tbl['card'].nda[ii]      = card
        tbl['channel'].nda[ii]   = channel
        tbl['fcio_id'].nda[ii]   = fcio_id
        tbl['numtraces'].nda[ii] = 1

        # set the time offsets
        offset = orca_header_length
        tbl['to_mu_sec'].nda[ii]      = np.int32(packet[offset])
        tbl['to_mu_usec'].nda[ii]     = np.int32(packet[offset+1])
        tbl['to_master_sec'].nda[ii]  = np.int32(packet[offset+2])
        tbl['to_dt_mu_usec'].nda[ii]  = np.int32(packet[offset+3])
        tbl['to_abs_mu_usec'].nda[ii] = np.int32(packet[offset+4])
        tbl['to_start_sec'].nda[ii]   = np.int32(packet[offset+5])
        tbl['to_start_usec'].nda[ii]  = np.int32(packet[offset+6])
        toff = np.float64(packet[offset+2]) + np.float64(packet[offset+3])*1e-6

        # set the dead region values
        offset += 7
        tbl['dr_start_pps'].nda[ii]   = np.int32(packet[offset])
        tbl['dr_start_ticks'].nda[ii] = np.int32(packet[offset+1])
        tbl['dr_stop_pps'].nda[ii]    = np.int32(packet[offset+2])
        tbl['dr_stop_ticks'].nda[ii]  = np.int32(packet[offset+3])
        tbl['dr_maxticks'].nda[ii]    = np.int32(packet[offset+4])

        # set the event number and clock counters
        offset += 5
        tbl['eventnumber'].nda[ii]         = np.int32(packet[offset])
        tbl['ts_pps'].nda[ii]       = np.int32(packet[offset+1])
        tbl['ts_ticks'].nda[ii]     = np.int32(packet[offset+2])
        tbl['ts_maxticks'].nda[ii]  = np.int32(packet[offset+3])

        # set the runtime and timestamp
        tstamp = np.float64(packet[offset+1]);
        tstamp += np.float64(packet[offset+2]) / (np.int32(packet[offset+3])+1)
        tbl['runtime'].nda[ii]   = tstamp
        tbl['timestamp'].nda[ii] = tstamp + toff

        # set the fpga baseline/energy and waveform
        offset = orca_header_length + fcio_header_length
        tbl['baseline'].nda[ii] =  packet[offset-1] & 0x0000ffff
        tbl['daqenergy'].nda[ii]   = (packet[offset-1] & 0xffff0000) >> 16
        wf = np.frombuffer(packet, dtype=np.uint16)[offset*2:
                                                    offset*2 + wf_samples]

        tbl['waveform']['values'].nda[ii][:wf_samples] = wf

        evt_rbkd[ccc].loc += 1
        return evt_rbkd[ccc].is_full()
