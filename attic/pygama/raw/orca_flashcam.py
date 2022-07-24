import copy
import gc
import logging
from typing import Any

import numpy as np

from pygama.raw.fc.fc_event_decoder import fc_decoded_values
from pygama.raw.orca.orca_base import OrcaDecoder
from pygama.raw.orca.orca_header import OrcaHeader
from pygama.raw.orca.orca_packet import OrcaPacket
from pygama.raw.raw_buffer import RawBufferLibrary

log = logging.getLogger(__name__)


class ORFlashCamListenerStatusDecoder(OrcaDecoder):
    """
    Decoder for FlashCam status packets written by ORCA

    Some of the card level status data contains an  array of values
    (temperatures for instance) for each card.  Since lh5 currently only
    supports a 1d vector of 1d vectors, this (card,value) data has to be
    flattened before populating the lh5 table.
    """

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
