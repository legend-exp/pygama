import logging
import os

import fcutils
import numpy as np

from pygama import lgdo

from ..data_streamer import DataStreamer
from ..raw_buffer import *
from .fc_config_decoder import FCConfigDecoder
from .fc_event_decoder import FCEventDecoder
from .fc_status_decoder import FCStatusDecoder

log = logging.getLogger(__name__)


class FCStreamer(DataStreamer):
    """
    decode FlashCam data, using the fcutils package to handle file access,
    and the FlashCam DataDecoders to save the results and write to output.
    """
    def __init__(self):
        super().__init__()
        self.fcio = None
        self.config_decoder = FCConfigDecoder()
        self.status_decoder = FCStatusDecoder()
        self.event_decoder = FCEventDecoder()
        self.event_tables = {}
        self.event_rbkd = None
        self.status_rb = None



    def get_decoder_list(self):
        dec_list = []
        dec_list.append(self.config_decoder)
        dec_list.append(self.status_decoder)
        dec_list.append(self.event_decoder)
        return dec_list



    def open_stream(self, fcio_filename, rb_lib=None, buffer_size=8192,
                    chunk_mode='any_full', out_stream=''):
        """ Initialize the FC data stream

        Parameters
        ----------
        fcio_filename : str
            the FCIO filename
        rb_lib : RawBufferLibrary
            library of buffers for this stream
        buffer_size : int
            length of tables to be read out in read_chunk
        chunk_mode : str
            DOCME
        out_stream : str
            DOCME

        Returns
        -------
        header_data : list(RawBuffer)
            a list of length 1 containing the raw buffer holding the fc_config table
        """
        self.fcio = fcutils.fcio(fcio_filename)
        self.n_bytes_read = 0

        # read in file header (config) info
        fc_config = self.config_decoder.decode_config(self.fcio) # returns an lgdo.Struct
        self.event_decoder.set_file_config(fc_config)
        self.n_bytes_read += 11*4 # there are 11 ints in the fcio_config struct

        # initialize the buffers in rb_lib. Store them for fast lookup
        super().open_stream(fcio_filename, rb_lib, buffer_size=buffer_size,
                            chunk_mode=chunk_mode, out_stream=out_stream)
        if rb_lib is None: rb_lib = self.rb_lib

        # get the status rb_list and pull out its first element
        status_rb_list = rb_lib['FCStatusDecoder'] if 'FCStatusDecoder' in rb_lib else None
        if status_rb_list is not None:
            if len(status_rb_list) != 1:
                log.warning(f'status_rb_list had length {len(status_rb_list)}, ignoring all but the first')
            if len(status_rb_list) == 0:
                self.status_rb = None
            else: self.status_rb = status_rb_list[0]
        self.event_rbkd = rb_lib['FCEventDecoder'].get_keyed_dict() if 'FCEventDecoder' in rb_lib else None

        # set up data loop variables
        self.packet_id = 0 # for storing packet order in output tables

        if 'FCConfigDecoder' in rb_lib:
            config_rb_list = rb_lib['FCConfigDecoder']
            if len(config_rb_list) != 1:
                log.warning(f'config_rb_list had length {len(config_rb_list)}, ignoring all but the first')
            rb = config_rb_list[0]
        else: rb = RawBuffer(lgdo=fc_config)
        rb.loc = 1 # we have filled this buffer
        return [rb]


    def close_stream(self):
        self.fcio = None # should cause close file in fcio.__dealloc__


    def read_packet(self):
        """ Read a packet of data.

        Data written to self.rb_lib.
        Updates self.n_bytes_read
        """

        rc = self.fcio.get_record()
        if rc == 0: return False # no more data

        self.packet_id += 1

        if rc == 1: # config (header) data
            log.warning(f'got a header after start of run? n_bytes_read = {self.n_bytes_read}')
            self.n_bytes_read += 11*4 # there are 11 ints in the fcio_config struct
            return True

        elif rc == 2: # calib record -- no longer supported
            log.warning(f'warning: got a calib record? n_bytes_read = {self.n_bytes_read}')
            return True

        # FIXME: push to a buffer of skipped packets?
        # FIXME: need to at least update n_bytes?
        elif rc == 5: # recevent
            log.warning(f'got a RecEvent packet -- skipping? n_bytes_read = {self.n_bytes_read}')
            # sizeof(fcio_recevent): (6 + 3*10 + 1*2304 + 3*4000)*4
            self.n_bytes_read += 57360
            return True

        # Status record
        elif rc == 4:
            if self.status_rb is not None:
                self.any_full |= self.status_decoder.decode_packet(self.fcio,
                                                                   self.status_rb,
                                                                   self.packet_id)
            # sizeof(fcio_status): (3 + 10 + 256*(10 + 9 + 16 + 4 + 256))*4
            self.n_bytes_read += 302132
            return True

        # Event or SparseEvent record
        elif rc == 3 or rc == 6:
            if self.event_rbkd is not None:
                self.any_full |= self.event_decoder.decode_packet(self.fcio,
                                                                  self.event_rbkd,
                                                                  self.packet_id)
            # sizeof(fcio_event): (5 + 3*10 + 1)*4 + numtraces*(1 + nsamples+2)*2
            self.n_bytes_read += 144 + self.fcio.numtraces*(self.fcio.nsamples + 3)*2
            return True
