import copy
import gc
import logging

import numpy as np

from ..fc.fc_event_decoder import fc_decoded_values
from .orca_base import OrcaDecoder

log = logging.getLogger(__name__)


def get_key(fcid, ch): return (fcid-1)*1000 + ch

def get_fcid(key): return int(np.floor(key/1000))+1

def get_ch(key): return key % 1000



class ORFlashCamListenerConfigDecoder(OrcaDecoder):
    '''
    Decoder for FlashCam listener config written by ORCA
    '''
    def __init__(self, header=None, **kwargs):
        """
        DOCME
        """
        # up through ch_inputnum, these are in order of the fcio data format
        # for similicity.  append any additional values after this.
        self.decoded_values = {
            'readout_id':   { 'dtype': 'uint16', },
            'fcid':         { 'dtype': 'uint16', },
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
        super().__init__(header=header, **kwargs)


    def get_decoded_values(self, key=None):
        return self.decoded_values


    def decode_packet(self, packet, packet_id, rbl):
        if len(rbl) != 1:
            print(f"FC config decoder: got {len(rbl)} rb's, should have only 1 (no keyed decoded values)")
        rb = rbl[0]
        tbl = rb.lgdo
        ii  = rb.loc

        int_packet = packet.astype(np.int32)
        tbl['readout_id'].nda[ii]  = (int_packet[1] & 0xffff0000) >> 16
        tbl['fcid'].nda[ii] =  int_packet[1] & 0x0000ffff

        for i,k in enumerate(self.decoded_values):
            if i < 2: continue
            tbl[k].nda[ii] = int_packet[i]
            if k == 'gps': break

        packet = packet[list(self.decoded_values.keys()).index('ch_boardid'):]
        for i in range(len(packet)):
            tbl['ch_boardid'].nda[ii][i]  = (packet[i] & 0xffff0000) >> 16
            tbl['ch_inputnum'].nda[ii][i] =  packet[i] & 0x0000ffff

        # check that the ADC decoder has the right number of samples
        objs = []
        for obj in gc.get_objects():
            try: 
                if isinstance(obj, ORFlashCamADCWaveformDecoder): objs.append(obj)
            except ReferenceError: 
                # avoids "weakly-referenced object no longer exists"
                pass
        if len(objs) != 1:
            log.warning(f'Got {len(objs)} ORFlashCamADCWaveformDecoders in memory!')
        else: objs[0].assert_nsamples(tbl['nsamples'].nda[ii], tbl['fcid'].nda[ii])

        rb.loc += 1
        return rb.is_full()



class ORCAFlashCamListenerStatusDecoder(OrcaDecoder):
    """
    Decoder for FlashCam status packets written by ORCA

    Some of the card level status data contains an  array of values
    (temperatures for instance) for each card.  Since lh5 currently only
    supports a 1d vector of 1d vectors, this (card,value) data has to be
    flattened before populating the lh5 table.
    """

    # def __init__(self, *args, **kwargs):

    #     self.decoder_name    = 'ORFlashCamListenerStatusDecoder'
    #     self.orca_class_name = 'ORFlashCamListenerModel'
    #     self.nOtherErrors    = np.uint32(5)
    #     self.nEnvMonitors    = np.uint32(16)
    #     self.nCardTemps      = np.uint32(5)
    #     self.nCardVoltages   = np.uint32(6)
    #     self.nADCTemps       = np.uint32(2)
    #     self.nCTILinks       = np.uint32(4)
    #     self.nCards          = np.uint32(1)

    #     self.decoded_values = {
    #         'readout_id':  { 'dtype': 'uint16', },
    #         'listener_id': { 'dtype': 'uint16', },
    #         'cards':       { 'dtype': 'int32',  },
    #         'status':      { 'dtype': 'int32',  },
    #         'statustime':  { 'dtype': 'float64', 'units': 's', },
    #         'cputime':     { 'dtype': 'float64', 'units': 's', },
    #         'startoffset': { 'dtype': 'float64', 'units': 's', },
    #         'card_fcio_id':  {
    #             'dtype':        'uint32',
    #             'datatype':     'array<1>{array<1>{real}}',
    #             'length_guess':  self.nCards, },
    #         'card_status': {
    #             'dtype':        'uint32',
    #             'datatype':     'array<1>{array<1>{real}}',
    #             'length_guess':  self.nCards, },
    #         'card_event_number': {
    #             'dtype':        'uint32',
    #             'datatype':     'array<1>{array<1>{real}}',
    #             'length_guess':  self.nCards, },
    #         'card_pps_count': {
    #             'dtype':        'uint32',
    #             'datatype':     'array<1>{array<1>{real}}',
    #             'length_guess':  self.nCards, },
    #         'card_tick_count': {
    #             'dtype':        'uint32',
    #             'datatype':     'array<1>{array<1>{real}}',
    #             'length_guess':  self.nCards, },
    #         'card_max_ticks': {
    #             'dtype':        'uint32',
    #             'datatype':     'array<1>{array<1>{real}}',
    #             'length_guess':  self.nCards, },
    #         'card_total_errors': {
    #             'dtype':        'uint32',
    #             'datatype':     'array<1>{array<1>{real}}',
    #             'length_guess':  self.nCards, },
    #         'card_env_errors': {
    #             'dtype':        'uint32',
    #             'datatype':     'array<1>{array<1>{real}}',
    #             'length_guess':  self.nCards, },
    #         'card_cti_errors': {
    #             'dtype':        'uint32',
    #             'datatype':     'array<1>{array<1>{real}}',
    #             'length_guess':  self.nCards, },
    #         'card_link_errors': {
    #             'dtype':        'uint32',
    #             'datatype':     'array<1>{array<1>{real}}',
    #             'length_guess':  self.nCards, },
    #         'card_other_errors': {
    #             'dtype':        'uint32',
    #             'datatype':     'array<1>{array<1>{real}}',
    #             'length_guess':  self.nCards * self.nOtherErrors, },
    #         'card_temp': {
    #             'dtype':        'uint32',
    #             'datatype':     'array<1>{array<1>{real}}',
    #             'length_guess':  self.nCards * self.nCardTemps,
    #             'units':        'mC', },
    #         'card_voltage': {
    #             'dtype':        'uint32',
    #             'datatype':     'array<1>{array<1>{real}}',
    #             'length_guess':  self.nCards * self.nCardVoltages,
    #             'units':        'mV', },
    #         'card_current': {
    #             'dtype':        'uint32',
    #             'datatype':     'array<1>{array<1>{real}}',
    #             'length_guess':  self.nCards,
    #             'units':        'mA', },
    #         'card_humidity': {
    #             'dtype':        'uint32',
    #             'datatype':     'array<1>{array<1>{real}}',
    #             'length_guess':  self.nCards,
    #             'units':        'o/oo', },
    #         'card_adc_temp': {
    #             'dtype':        'uint32',
    #             'datatype':     'array<1>{array<1>{real}}',
    #             'length_guess':  self.nCards * self.nADCTemps,
    #             'units':        'mC', },
    #         'card_cti_link': {
    #             'dtype':        'uint32',
    #             'datatype':     'array<1>{array<1>{real}}',
    #             'length_guess':  self.nCards * self.nCTILinks, },
    #         'card_card_link_state': {
    #             'dtype':        'uint32',
    #             'datatype':     'array<1>{array<1>{real}}',
    #             'length_guess':  self.nCards * self.nCards, },
    #     }

    #     # arrays to temporarily store card-level decoded data
    #     self.cdata = {}
    #     self.resize_card_data(ncards=self.nCards)

    #     super().__init__(args, kwargs)


    # def resize_card_data(self, ncards):
    #     try: ncards = np.uint32(ncards)
    #     except ValueError: return
    #     if ncards == 0: return
    #     for key in self.decoded_values:
    #         # ignore keys that aren't card level
    #         if key.find('card_') != 0: continue
    #         try:
    #             # skip keys for things that aren't arrays with a length_guess
    #             if self.decoded_values[key]['datatype'].find('array') != 0:
    #                 continue
    #             length = self.decoded_values[key]['length_guess']
    #             try:
    #                 # resize if ncards differs from the old shape
    #                 oldshape = self.cdata[key].shape
    #                 if oldshape[0] == ncards: continue
    #                 if key.find('card_card_') == 0:
    #                     self.cdata[key].resize((ncards,ncards,) + oldshape[2:])
    #                 else:
    #                     self.cdata[key].resize((ncards,) + oldshape[1:])
    #             except KeyError:
    #                 # if the key didn't exist set the ndarray for this key
    #                 if ((length == ncards or (length % ncards) != 0) and
    #                     key.find('card_card_') == -1):
    #                     self.cdata[key] = np.ndarray(shape=(length),
    #                                                  dtype=np.uint32)
    #                 else:
    #                     nval = np.uint32(length / ncards)
    #                     self.cdata[key] = np.ndarray(shape=(ncards, nval),
    #                                                  dtype=np.uint32)
    #         except KeyError: continue
    #     # set nCards to allow for not calling this function during decoding
    #     self.nCards = ncards


    # def get_decoded_values(self, key=None):
    #     return self.decoded_values


    # def max_n_rows_per_packet(self):
    #     return 1


    # def decode_packet(self, packet, lh5_tables, packet_id, header_dict):

    #     data = np.frombuffer(packet, dtype=np.uint32)
    #     tbl  = lh5_tables
    #     ii   = tbl.loc

    #     # populate the packet header information
    #     tbl['readout_id'].nda[ii]  = (data[0] & 0xffff0000) >> 16
    #     tbl['listener_id'].nda[ii] =  data[0] & 0x0000ffff
    #     tbl['status'].nda[ii]      = np.int32(data[1])
    #     tbl['statustime'].nda[ii]  = np.float64(data[2] + data[3] / 1.0e6)
    #     tbl['cputime'].nda[ii]     = np.float64(data[4] + data[5] / 1.0e6)
    #     tbl['startoffset'].nda[ii] = np.float64(data[7] + data[8] / 1.0e6)
    #     tbl['cards'].nda[ii]       = np.int32(data[12])

    #     # resize the card level data if necessary
    #     if data[12] != self.nCards:
    #         print('ORlashCamListenerStatusDecoder: resizing card arrays '
    #               'from', self.nCards, ' cards to', data[12])
    #         self.resize_card_data(ncards=data[12])

    #     # set the local card level data
    #     for i in range(np.int(data[12])):
    #         j = 14 + i * (data[12] + 14 + self.nOtherErrors +
    #                       self.nEnvMonitors + self.nCTILinks)

    #         self.cdata['card_fcio_id'][i]      = data[j]
    #         self.cdata['card_status'][i]       = data[j+1]
    #         self.cdata['card_event_number'][i] = data[j+2]
    #         self.cdata['card_pps_count'][i]    = data[j+3]
    #         self.cdata['card_tick_count'][i]   = data[j+4]
    #         self.cdata['card_max_ticks'][i]    = data[j+5]
    #         self.cdata['card_total_errors'][i] = data[j+10]
    #         self.cdata['card_env_errors'][i]   = data[j+11]
    #         self.cdata['card_cti_errors'][i]   = data[j+12]
    #         self.cdata['card_link_errors'][i]  = data[j+13]
    #         k = j + 14
    #         self.cdata['card_other_errors'][i][:]= data[k:k+self.nOtherErrors]
    #         k += self.nOtherErrors
    #         self.cdata['card_temp'][i][:]        = data[k:k+self.nCardTemps]
    #         k += self.nCardTemps
    #         self.cdata['card_voltage'][i][:]     = data[k:k+self.nCardVoltages]
    #         k += self.nCardVoltages
    #         self.cdata['card_current'][i]        = data[k]
    #         self.cdata['card_humidity'][i]       = data[k+1]
    #         k += 2
    #         self.cdata['card_adc_temp'][i][:]    = data[k:k+self.nADCTemps]
    #         k += self.nADCTemps
    #         self.cdata['card_cti_link'][i][:]    = data[k:k+self.nCTILinks]
    #         k += self.nCTILinks
    #         self.cdata['card_card_link_state'][i][:]  = data[k:k+data[12]]

    #     # populate the card level data with the flattened local data, then push
    #     for key in self.cdata:
    #         tbl[key].set_vector(ii, self.cdata[key].flatten())

    #     tbl.push_row()

class ORFlashCamADCWaveformDecoder(OrcaDecoder):
    """
    Decoder for FlashCam ADC data written by ORCA
    """
    def __init__(self, header=None, **kwargs):
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
            'fcid' :    { 'dtype': 'uint8',  },
        } )
        self.decoded_values = {} # dict[fcid]
        self.fcid = {} # dict[crate][card]
        self.nadc = {} # dict[fcid]
        super().__init__(header=header, **kwargs)
        self.skipped_channels = {}


    def set_header(self, header):
        self.header = header

        # set up decoded values, key list, fcid map, etc. based on header info
        fc_listener_info = header.get_readout_info('ORFlashCamListenerModel')
        for info in fc_listener_info:
            fcid = info['uniqueID']
            # we are going to subtract 1 from fcid for the keys, so it better start from 1
            if fcid == 0: raise ValueError("got fcid=0 unexpectedly!")

            obj_info_dict = header.get_object_info('ORFlashCamADCModel')
            self.nadc[fcid] = 0
            for child in info['children']:
                # load self.fcid
                crate = child['crate']
                if crate not in self.fcid: self.fcid[crate] = {}
                card = child['station']
                self.fcid[crate][card] = fcid

                # load self.nadc
                self.nadc[fcid] += np.count_nonzero(obj_info_dict[crate][card]['Enabled'])

            # we are going to shift by 1000 for each fcid, so we better not have that many adcs!
            if self.nadc[fcid] > 1000: raise ValueError(f"got too many adc's! ({nadc})")

            # get the wf len for this fcid and set up decoded_values
            wf_len = header.get_auxhw_info('ORFlashCamListenerModel', fcid)['eventSamples']
            self.decoded_values[fcid] = copy.deepcopy(self.decoded_values_template)
            self.decoded_values[fcid]['waveform']['wf_len'] = wf_len


    def get_key_list(self):
        key_list = []
        for fcid, nadc in self.nadc.items():
            key_list += list( get_key(fcid, np.array(range(nadc))) )
        return key_list


    def get_decoded_values(self, key=None):
        if key is None:
            dec_vals_list = self.decoded_values.values()
            if len(dec_vals_list) >= 0: return dec_vals_list[0]
            raise RuntimeError('decoded_values not built')
        fcid = get_fcid(key)
        if fcid in self.decoded_values: return self.decoded_values[fcid]
        raise KeyError(f'no decoded values for key {key} (fcid {fcid})')


    def assert_nsamples(self, nsamples, fcid):
        orca_nsamples = self.decoded_values[fcid]['waveform']['wf_len']
        if orca_nsamples != nsamples:
            log.warning(f"orca miscalculated nsamples = {orca_nsamples} for fcid {fcid}, updating to {nsamples}")
            self.decoded_values[fcid]['waveform']['wf_len'] = nsamples


    def decode_packet(self, packet, packet_id, rbl):
        ''' decode the orca FC ADC packet '''
        evt_rbkd = rbl.get_keyed_dict()

        # unpack lengths and ids from the header words
        orca_header_length = (packet[1] & 0xf0000000) >> 28
        fcio_header_length = (packet[1] & 0x0fc00000) >> 22
        wf_samples         = (packet[1] & 0x003fffc0) >> 6
        crate              = (packet[2] & 0xf8000000) >> 27
        card               = (packet[2] & 0x07c00000) >> 22
        fcid = self.fcid[crate][card]
        ch_orca            = (packet[2] & 0x00003c00) >> 10
        channel            =  packet[2] & 0x000003ff
        key = get_key(fcid, channel)

        # get the table for this crate/card/channel
        if key not in evt_rbkd:
            if key not in self.skipped_channels:
                self.skipped_channels[key] = 0
            self.skipped_channels[key] += 1
            return False
        tbl = evt_rbkd[key].lgdo
        ii = evt_rbkd[key].loc

        # check that the waveform length is as expected
        rb_wf_len = tbl['waveform']['values'].nda.shape[1]
        if wf_samples != rb_wf_len:
            if not hasattr(self, 'wf_len_errs'): self.wf_len_errs = {}
            # if dec_vals has been updated, orca miscalc'd and a warning has
            # already been emitted.  Otherwise, emit a new warning.
            if wf_samples != self.decoded_values[fcid]['waveform']['wf_len']:
                if fcid not in self.wf_len_errs:
                    log.warning(f'got waveform from fcid {fcid} of length {wf_samples} with expected length {rb_wf_len}')
                    self.wf_len_errs[fcid] = True
            # Now resize buffer only if it is still empty.
            # Otherwise emit a warning and keep the smaller length
            if ii != 0:
                if fcid not in self.wf_len_errs:
                    log.warning(f'tried to resize buffer according to config record but it was not empty!')
                    self.wf_len_errs[fcid] = True
                if wf_samples > rb_wf_len: wf_samples = rb_wf_len
            else: tbl['waveform'].resize_wf_len(wf_samples)

        # set the values decoded from the header words
        tbl['packet_id'].nda[ii] = packet_id
        tbl['crate'].nda[ii]     = crate
        tbl['card'].nda[ii]      = card
        tbl['ch_orca'].nda[ii]   = ch_orca
        tbl['channel'].nda[ii]   = channel
        tbl['fcid'].nda[ii]      = fcid
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

        evt_rbkd[key].loc += 1
        return evt_rbkd[key].is_full()
