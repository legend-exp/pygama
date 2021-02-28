import os
import numpy as np
from pprint import pprint
from collections import defaultdict

from ..utils import *
from .io_base import DataDecoder
from pygama import lh5
from .ch_group import *


def process_flashcam(daq_file, raw_files, n_max, ch_groups_dict=None, verbose=False, buffer_size=8192, chans=None, f_out = ''):
    """
    decode FlashCam data, using the fcutils package to handle file access,
    and the FlashCam DataTaker to save the results and write to output.

    `raw_files` can be a string, or a dict with a label for each file:
      `{'geds':'filename_geds.lh5', 'muvt':'filename_muvt.lh5}`
    """
    import fcutils

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

    # build ch_groups and set up tables
    ch_groups = None
    if (ch_groups_dict is not None) and ('FlashCamEventDecoder' in ch_groups_dict):
        # get ch_groups
        ch_groups = ch_groups_dict['FlashCamEventDecoder']
        expand_ch_groups(ch_groups)
    else:
        print('Config not found.  Single-table mode')
        ch_groups = create_dummy_ch_group()

    # set up ch_group-to-output-file-and-group info
    if single_output:
        set_outputs(ch_groups, out_file_template=f_out, grp_path_template='{group_name}/raw')
    else:
        set_outputs(ch_groups, out_file_template=raw_files, grp_path_template='{group_name}/raw')

    # set up tables
    event_tables = build_tables(ch_groups, buffer_size, event_decoder)

    if verbose:
        print('Output group : output file')
        for group_info in ch_groups.values():
            group_path = group_info['group_path']
            out_file = group_info['out_file']
            print(group_path, ':', out_file.split('/')[-1])

    # set up status decoder (this is 'auxs' output)
    status_decoder = FlashCamStatusDecoder()
    status_decoder.set_file_config(fcio)
    status_tbl = lh5.Table(buffer_size)
    status_decoder.initialize_lh5_table(status_tbl)
    try:
      status_filename = f_out if single_output else raw_files['auxs']
      config_filename = f_out if single_output else raw_files['auxs']
    except:
      status_filename = "fcio_status"
      config_filename = "fcio_config"

    # Set up the store
    # TODO: add overwrite capability
    lh5_store = lh5.Store()

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
            for group_info in ch_groups.values():
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
                    tbl.clear()

            # Looks okay: just decode
            bytes_processed += event_decoder.decode_packet(fcio, event_tables, packet_id)

            # i_debug += 1
            # if i_debug == 10:
            #    print("breaking early")
            #    break # debug, deleteme

    # end of loop, write to file once more
    for group_info in ch_groups.values():
        tbl = group_info['table']
        if tbl.loc != 0:
            group_path = group_info['group_path']
            out_file = group_info['out_file']
            lh5_store.write_object(tbl, group_path, out_file, n_rows=tbl.loc)
            tbl.clear()
    if status_tbl.loc != 0:
        lh5_store.write_object(status_tbl, 'stat', status_filename,
                               n_rows=status_tbl.loc)
        status_tbl.clear()

    if verbose:
        update_progress(1)
        print(packet_id, 'packets decoded')

    if len(event_decoder.skipped_channels) > 0:
        print("Warning - daq_to_raw skipped some channels in file")
        if verbose:
            for ch, n in event_decoder.skipped_channels.items():
                print("  ch", ch, ":", n, "hits")

    return bytes_processed
