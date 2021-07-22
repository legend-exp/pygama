import os
import numpy as np
from pygama import lgdo
from ..raw_groups import *
from ..data_streamer import DataStreamer
from .fc_config_decoder import FCConfigDecoder
from .fc_event_decoder import FCEventDecoder
from .fc_status_decoder import FCStatusDecoder
import fcutils

class FCStreamer(DataStreamer):
    """
    decode FlashCam data, using the fcutils package to handle file access,
    and the FlashCam DataDecoders to save the results and write to output.
    """
    def __init__(self):
        super().__init__(self)
        self.config_decoder = FCConfigDecoder()
        self.status_decoder = FCStatusDecoder()
        self.event_decoder = FCEventDecoder()
        self.event_tables = {}

    def get_decoder_list(self):
        dec_list = []
        dec_list.append(self.config_decoder)
        dec_list.append(self.status_decoder)
        dec_list.append(self.event_decoder)

    def initialize(self, fcio_file, rb_lib, buffer_size=8192, verbosity=0):
        """ Initialize the FC data stream

        Parameters
        ----------
        fcio_file : str
            the FCIO filename 
        rb_lib : RawBufferLibrary
            library of buffers for this stream
        buffer_size : int
            length of tables to be read out in read_chunk
        verbosity : int
            verbosity level for the initialize function
 
        Returns
        -------
        header_data, n_bytes : list(RawBuffer), int
            header_data is a list of length 1 containing the raw buffer holding
                the fc_config table
            n_bytes is the number of bytes read from the file to extract the
                header data
        """
        self.fcio = fcutils.fcio(fcio_file)

        # read in file header (config) info
        fc_config = self.config_decoder.decode_config(fcio)
        self.event_decoder.set_file_config(fc_config)

        super().initialize(fcio_file, rb_lib, buffer_size=buffer_size, verbosity=verbosity)

        # build raw_groups and set up tables
        self.raw_groups = None
        if (raw_groups_library is not None) and ('FCEventDecoder' in raw_groups_dict):
            # get raw_groups
            self.raw_groups = raw_groups_dict['FCEventDecoder']
            expand_raw_groups(self.raw_groups)
        else:
            if verbosity > 0: print('Config not found.  Single-table mode')
            self.raw_groups = create_dummy_raw_groups()
        self.event_tables = build_tables(self.raw_groups, buffer_size, init_obj=event_decoder)

        self.status_tbl = lgdo.Table(buffer_size)
        self.status_decoder.initialize_lgdo_table(status_tbl)

        # set up data loop variables
        self.packet_id = 0
        self.max_numtraces = 0
        return [fc_config]



    def read_chunk(self, verbosity=0):
        """
        Returns
        -------
        chunk : list of lgdos
            the returned list of lgdos (tables, etc) is ready for writing to
            file or further processing
        """
        rc = 1
        n_bytes = 0
        while rc:
            rc = self.fcio.get_record()

            # Skip non-interesting records
            # FIXME: push to a buffer of skipped packets?
            if rc == 0 or rc == 1 or rc == 2 or rc == 5: continue

            packet_id += 1

            # Status record
            if rc == 4:
                n_bytes += self.status_decoder.decode_packet(self.fcio, self.status_tbl, self.packet_id)
                if self.status_tbl.is_full(): return [ self.status_tbl ], n_bytes

            # Event or SparseEvent record
            if rc == 3 or rc == 6:

                # check that tables are large enough to read in this packet. If
                # not, exit and return the ones that might overflow
                full_tables = []
                for group_info in self.raw_groups.values():
                    tbl = group_info['table']
                    # Check that the tables are large enough
                    # TODO: don't need to check this every event, only if sum(numtraces) >= buffer_size
                    if tbl.size < self.fcio.numtraces and self.fcio.numtraces > self.max_numtraces:
                        print('warning: tbl.size =', tbl.size,
                              'but fcio.numtraces =', self.fcio.numtraces)
                        print('may overflow. suggest increasing tbl.size')
                        self.max_numtraces = fcio.numtraces
                    # Return if tables are too full.
                    if tbl.size - tbl.loc < fcio.numtraces: full_tables.append(tbl)
                if len(full_tables) > 0: return full_tables, n_bytes

                # Looks okay: just decode
                n_bytes += self.event_decoder.decode_packet(self.fcio, self.event_tables, self.packet_id)

    # finished with loop. return any tables with data
    tables = []
    for group_info in self.raw_groups.values():
        tbl = group_info['table']
        if tbl.loc != 0: tables.append(tbl)
    if self.status_tbl.loc != 0: tables.append(tbl)
    if len(full_tables) > 0: return tables, n_bytes


def process_flashcam(daq_file, raw_files, n_max, raw_groups_dict=None, verbose=False, buffer_size=8192, chans=None, f_out = ''):
    """
    `raw_files` can be a string, or a dict with a label for each file:
      `{'geds':'filename_geds.lh5', 'muvt':'filename_muvt.lh5}`
    """
    if isinstance(raw_files, str):
        single_output = True
        f_out = raw_files
    elif len(raw_files) == 1:
        single_output = True
        f_out = list(raw_files.values())[0]
    else:
        single_output = False

    fcio = fcutils.fcio(daq_file)

    # set up event decoder
    event_decoder = FlashCamEventDecoder()
    event_decoder.set_file_config(fcio)
    event_tables = {}

    # build raw_groups and set up tables
    raw_groups = None
    if (raw_groups_dict is not None) and ('FlashCamEventDecoder' in raw_groups_dict):
        # get raw_groups
        raw_groups = raw_groups_dict['FlashCamEventDecoder']
        expand_raw_groups(raw_groups)
    else:
        print('Config not found.  Single-table mode')
        raw_groups = create_dummy_raw_groups()

    # set up raw_group-to-output-file-and-group info
    if single_output:
        set_outputs(raw_groups, out_file_template=f_out, grp_path_template='{group_name}/raw')
    else:
        set_outputs(raw_groups, out_file_template=raw_files, grp_path_template='{group_name}/raw')

    # set up tables
    event_tables = build_tables(raw_groups, buffer_size, event_decoder)

    if verbose:
        print('Output group : output file')
        for group_info in raw_groups.values():
            group_path = group_info['group_path']
            out_file = group_info['out_file']
            print(group_path, ':', out_file.split('/')[-1])

    # dictionary with the unique file names as keys
    file_info = dict.fromkeys(set(group_info['out_file'] for group_info in ch_groups.values()), False)

    # set up status decoder (this is 'auxs' output)
    status_decoder = FlashCamStatusDecoder()
    status_decoder.set_file_config(fcio)
    status_tbl = lgdo.Table(buffer_size)
    status_decoder.initialize_lgdo_table(status_tbl)
    try:
      status_filename = f_out if single_output else raw_files['auxs']
      config_filename = f_out if single_output else raw_files['auxs']
    except:
      status_filename = "fcio_status"
      config_filename = "fcio_config"

    # Set up the store
    # TODO: add overwrite capability
    lh5_store = lgdo.LH5Store()

    # write fcio_config
    fcio_config = event_decoder.get_file_config_struct()
    lh5_store.write_object(fcio_config, 'fcio_config', config_filename)

    # loop over raw data packets
    i_debug = 0
    packet_id = 0
    rc = 1
    bytes_processed = 0
    file_size = os.path.getsize(daq_file)
    max_numtraces = 0
    while rc and packet_id < n_max:
        rc = fcio.get_record()

        # Skip non-interesting records
        # FIXME: push to a buffer of skipped packets?
        if rc == 0 or rc == 1 or rc == 2 or rc == 5: continue

        packet_id += 1

        if verbose and packet_id % 1000 == 0:
            # FIXME: is cast to float necessary?
            pct_done = bytes_processed / file_size
            if n_max < np.inf and n_max > 0: pct_done = packet_id / n_max
            update_progress(pct_done)

        # Status record
        if rc == 4:
            bytes_processed += status_decoder.decode_packet(fcio, status_tbl, packet_id)
            if status_tbl.is_full():
                lh5_store.write_object(status_tbl, 'fcio_status', status_filename, n_rows=status_tbl.size)
                status_tbl.clear()

        # Event or SparseEvent record
        if rc == 3 or rc == 6:
            for group_info in raw_groups.values():
                tbl = group_info['table']
                # Check that the tables are large enough
                # TODO: don't need to check this every event, only if sum(numtraces) >= buffer_size
                if tbl.size < fcio.numtraces and fcio.numtraces > max_numtraces:
                    print('warning: tbl.size =', tbl.size, 'but fcio.numtraces =', fcio.numtraces)
                    print('may overflow. suggest increasing tbl.size')
                    max_numtraces = fcio.numtraces
                # Pre-emptively clear tables if it might be necessary
                if tbl.size - tbl.loc < fcio.numtraces: # might overflow
                    group_path = group_info['group_path']
                    out_file = group_info['out_file']
                    lh5_store.write_object(tbl, group_path, out_file, n_rows=tbl.loc)
                    if out_file in file_info: file_info[out_file] = True
                    tbl.clear()

            # Looks okay: just decode
            bytes_processed += event_decoder.decode_packet(fcio, event_tables, packet_id)

            # i_debug += 1
            # if i_debug == 10:
            #    print("breaking early")
            #    break # debug, deleteme

    # end of loop, write to file once more
    for group_info in raw_groups.values():
        tbl = group_info['table']
        if tbl.loc != 0:
            group_path = group_info['group_path']
            out_file = group_info['out_file']
            lh5_store.write_object(tbl, group_path, out_file, n_rows=tbl.loc)
            if out_file in file_info: file_info[out_file] = True
            tbl.clear()
    if status_tbl.loc != 0:
        lh5_store.write_object(status_tbl, 'stat', status_filename,
                               n_rows=status_tbl.loc)
        status_tbl.clear()

    # alert user to any files not actually saved in the end
    for out_file, is_saved in file_info.items():
        if not is_saved:
            print('Not saving file since no data were found:', out_file)

    if verbose:
        update_progress(1)
        print(packet_id, 'packets decoded')

    if len(event_decoder.skipped_channels) > 0:
        print("Warning - daq_to_raw skipped some channels in file")
        if verbose:
            for ch, n in event_decoder.skipped_channels.items():
                print("  ch", ch, ":", n, "hits")

    return bytes_processed
