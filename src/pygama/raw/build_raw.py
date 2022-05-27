import glob
import json
import os
import sys
import time

import numpy as np
import tqdm

from pygama import lgdo
from pygama.math.utils import sizeof_fmt

from .fc.fc_streamer import FCStreamer
from .orca.orca_streamer import OrcaStreamer
from .raw_buffer import (
    RawBuffer,
    RawBufferLibrary,
    RawBufferList,
    write_to_lh5_and_clear,
)

#from stream_llama import *
#from stream_compass import *
#from stream_fc import *


def build_raw(in_stream, in_stream_type=None, out_spec=None, buffer_size=8192,
              n_max=np.inf, overwrite=True, verbosity=2, **kwargs):
    """ Convert data into LEGEND hdf5 `raw` format.

    Takes an input stream (in_stream) of a given type (in_stream_type) and
    writes to output file(s) according to the user's a specification (out_spec).

    Parameters
    ----------
    in_stream : str
        The name of the input stream to be converted. Typically a filename,
        including path. Can use environment variables. Some streamers may be
        able to (eventually) accept e.g. streaming over a port as an input.

    in_stream_type : str
        Type of stream used to write the input file.
        Options are 'ORCA', 'FlashCams', 'LlamaDaq', 'Compass', 'MGDO'

    out_spec : str or json dict or RawBufferLibrary or None
        Specification for the output stream.

        - If None, uses '{in_stream}.hdf5' as the output filename.
        - If a str not ending in '.json', interpreted as the output filename.
        - If a str ending in '.json', interpreted as a filename containing
          json-shorthand for the output specification (see raw_buffer.py)
        - If a json dict, should be a dict loaded from the json shorthand
          notation for RawBufferLibraries (see raw_buffer.py), which is then
          used to build a RawBufferLibrary
        - If a RawBufferLibrary, the mapping of data to output file / group is
          taken from that.

    buffer_size : int
        Default size to use for data buffering

    n_max : int
        Maximum number of "row" of data to process from the input file

    overwrite : bool
        Sets whether to overwrite the output file(s) if it (they) already exist

    verbosity : int
        Sets the verbosity level. 0 gives the minimum output level.

    **kwargs : kwargs
        Sent to RawBufferLibrary generation as kw_dict
    """

    # convert any environment variables in in_stream so that we can check for readability
    in_stream = os.path.expandvars(in_stream)
    # later: fix if in_stream is not a file
    # (e.g. a socket, which exists if open and has size = np.inf)
    if not os.path.exists(in_stream):
        print(f'Error: file {in_stream} not found')
        return
    in_stream_size = os.stat(in_stream).st_size

    # try to guess the input stream type if it's not provided
    if in_stream_type is None:
        i_ext = in_stream.rfind('.')
        if i_ext == -1:
            if OrcaStreamer.is_orca_stream(in_stream): in_stream_type = 'ORCA'
            else:
                print('unknown file type. Specify in_stream_type')
                return
        else:
            ext = in_stream[i_ext+1:]
            if ext == 'fcio': in_stream_type = 'FlashCam'
            if ext == 'gz' and OrcaStreamer.is_orca_stream(in_stream): in_stream_type = 'ORCA'
            else:
                print(f'unknown file extension {ext}. Specify in_stream_type')
                return

    # procss out_spec and setup rb_lib if specified
    rb_lib = None
    if isinstance(out_spec, str) and out_spec.endswith('.json'):
        with open(out_spec) as json_file: out_spec = json.load(json_file)
    if isinstance(out_spec, dict):
        out_spec = RawBufferLibrary(json_dict=out_spec, kw_dict=kwargs)
    if isinstance(out_spec, RawBufferLibrary): rb_lib = out_spec
    # if no rb_lib, write all data to file
    if out_spec is None:
        out_spec = in_stream
        i_ext = out_spec.rfind('.')
        if i_ext != -1: out_spec = out_spec[:i_ext]
        out_spec += '.lh5'
    # by now, out_spec should be a str or a RawBufferLibrary
    if not isinstance(out_spec, str) and not isinstance(out_spec, RawBufferLibrary):
        print(f'Error: unknown out_spec type {type(out_spec).__name__}')
        return

    # modify buffer_size if necessary for n_max
    if buffer_size < 1:
        print(f'Error: bad buffer_size {buffer_size}')
        return
    if buffer_size > n_max: buffer_size = n_max

    # output start of processing info if verbosity > 0
    if verbosity > 0:
        print( 'Starting build_raw processing.')
        print(f'  Input: {in_stream}')
        out_files = [out_spec]
        if isinstance(out_spec, RawBufferLibrary):
            out_files = out_spec.get_list_of('out_stream')
        if len(out_files) == 1: print(f'  Output: {out_files[0]}')
        else:
            print(f'  Output:')
            for out_file in out_files: print(f'- {out_file}')
        print(f'  Buffer size: {buffer_size}')
        print(f'  Max num. events: {n_max}')
        if verbosity > 1:
            if n_max < np.inf: progress_bar = tqdm.tqdm(total=n_max, unit='rows')
            else: progress_bar = tqdm.tqdm(total=in_stream_size, unit='B', unit_scale=True)

    # start a timer and a byte counter
    t_start = time.time()

    # select the appropriate streamer for in_stream
    streamer = None
    if in_stream_type == 'ORCA':
        streamer = OrcaStreamer()
    elif in_stream_type == 'FlashCam':
        streamer = FCStreamer()
    elif in_stream_type == 'LlamaDaq':
        print(f'Error: LlamaDaq streaming not yet implemented')
        return
    elif in_stream_type == 'Compass':
        print(f'Error: Compass streaming not yet implemented')
        return
    elif in_stream_type == 'MGDO':
        print(f'Error: MGDO streaming not yet implemented')
        return
    else:
        print(f'Error: Unknown input stream type {in_stream_type}')
        return

    # initialize the stream and read header. Also initializes rb_lib
    if verbosity > 1: progress_bar.update(0)
    out_stream = out_spec if isinstance(out_spec, str) else ''
    header_data = streamer.open_stream(in_stream, rb_lib=rb_lib, buffer_size=buffer_size,
                                       chunk_mode='full_only', out_stream=out_stream,
                                       verbosity=verbosity)
    rb_lib = streamer.rb_lib
    if verbosity > 1 and n_max == np.inf: progress_bar.update(streamer.n_bytes_read)

    # rb_lib should now be fully initialized. Check if files need to be
    # overwritten or if we need to stop to avoid overwriting
    out_files = rb_lib.get_list_of('out_stream')
    for out_file in out_files:
        colpos = out_file.find(':')
        if colpos != -1: out_file = out_file[:colpos]
        out_file_glob = glob.glob(out_file)
        if len(out_file_glob) == 0: continue
        if len(out_file_glob) > 1:
            print(f'Error: got multiple matches for out_file {out_file}: {out_file_glob}')
            return
        if not overwrite:
            print(f'Error: file {out_file_glob[0]} exists. Use option overwrite to proceed.')
            return
        os.remove(out_file_glob[0])

    # Write header data
    lh5_store = lgdo.LH5Store(keep_open=True)
    write_to_lh5_and_clear(header_data, lh5_store)

    # Now loop through the data
    n_bytes_last = streamer.n_bytes_read
    while True:
        chunk_list = streamer.read_chunk(verbosity=verbosity-2)
        if verbosity > 1 and n_max == np.inf:
            progress_bar.update(streamer.n_bytes_read-n_bytes_last)
            n_bytes_last = streamer.n_bytes_read
        if len(chunk_list) == 0: break
        n_read = 0
        for rb in chunk_list:
            if rb.loc > n_max: rb.loc = n_max
            n_max -= rb.loc
            n_read += rb.loc
        if verbosity > 1 and n_max < np.inf: progress_bar.update(n_read)
        write_to_lh5_and_clear(chunk_list, lh5_store)
        if n_max <= 0: break

    # --------- summary ------------

    if verbosity > 0:
        elapsed = time.time() - t_start
        print(f"Time elapsed: {elapsed:.2f} sec")
        out_files = rb_lib.get_list_of('out_stream')
        if len(out_files) == 1:
            out_file = out_files[0]
            colpos = out_file.find(':')
            if colpos != -1: out_file = out_file[:colpos]
            if os.path.exists(out_file):
                file_size = os.stat(out_file).st_size
                print(f"Output file: {out_file} ({sizeof_fmt(file_size)})")
            else: print("Output file: {out_file} (not written)")
        else:
            print("Output files:")
            for out_file in out_files:
                colpos = out_file.find(':')
                if colpos != -1: out_file = out_file[:colpos]
                if os.path.exists(out_file):
                    file_size = os.stat(out_file).st_size
                    print(f"  {out_file} ({sizeof_fmt(file_size)})")
                else: print(f"  {out_file} (not written)")
        print(f"Total converted: {sizeof_fmt(streamer.n_bytes_read)}")
        print(f"Conversion speed: {sizeof_fmt(streamer.n_bytes_read/elapsed)}ps")

        print('Done.\n')
