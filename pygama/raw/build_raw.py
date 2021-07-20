import os, time, sys, glob
import numpy as np

from pygama.math.utils import update_progress
from raw_buffer import RawBuffer, RawBufferList, RawBufferLibrary, write_to_lh5_and_clear
from fc.fc_streamer import FCStreamer

#from orca.stream_orca import *
#from stream_llama import *
#from stream_compass import *
#from stream_fc import *


def build_raw_files(in_stream, in_stream_type, out_spec, buffer_size=8192, 
                    n_max=np.inf, overwrite=True, verbosity=0)
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
        - If a str, interpretted as a filename for
          an output hdf5 file, in which each stream decoder writes its own
          stream to an hdf5 group named for the decoder. Can use environment
          variables.
        - If None, uses '{in_stream}.hdf5' as the output filename.  
        - If a RawBufferLibrary, the mapping of data to output file / group is
          taken from that.
        - If a json dict, should be a dict loaded from the json shorthand
          notation for RawBufferLibraries (see raw_buffer.py), which is then
          used to build a RawBufferLibrary
    buffer_size : int
        Default size to use for data buffering
    n_max : int
        Maximum number of "row" of data to process from the input file
    overwrite : bool
        Sets whether to overwrite the output file(s) if it (they) already exist
    verbosity : int
        Sets the verbosity level. 0 gives the minimum output level.
    """

    # convert any environment variables in in_stream so that we can check for readability
    in_stream = os.path.expandvars(in_stream)
    # later: fix if in_stream is not a file 
    # (e.g. a socket, which exists if open and has size = np.inf)
    if not os.path.exists(in_stream):
        print(f'Error: file {in_stream} not found')
        return
    in_stream_size = os.stat(in_stream).st_size 

    # set up RawBufferLibrary
    if isinstance(out_spec, RawBufferLibrary): rb_lib = out_spec
    elif isinstance(out_spec, dict): rb_lib = RawBufferLibrary(json_dict=out_spec)
    else: # dummy rb_lib sending all data to out_spec
        if out_spec is None: out_spec = f'{in_stream}.hdf5'
        rb_lib = RawBufferLibrary()
        rb_lib['*'] = RawBufferList()
        rb_lib['*'].append(RawBuffer(out_stream=out_spec, out_name='{name}'))

    # modify buffer_size if necessary for n_max
    if buffer_size < 1:
        print(f'Error: bad buffer_size {buffer_size}')
        return
    if buffer_size > n_max: buffer_size = n_max

    # ouput start of processing info if verbosity > 0
    if verbosity > 0:
        output = out_spec
        if isinstance(out_spec, RawBufferLibrary): 
            output = out_spec.get_list_of('out_stream')
        print( 'Starting build_raw processing.')
        print(f'  Input: {in_stream}')        
        print(f'  Output: {pprint(output)}')
        print(f'  Buffer size: {buffer_size}')  
        print(f'  Max num. events: {n_max}')    

    # start a timer and a byte counter
    t_start = time.time()
    bytes_processed = None

    # select the approprate streamer for in_stream
    streamer = None
    if in_stream_type == 'ORCA':
        print(f'Error: ORCA streaming not yet implemented')
        return
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
    header_data, n_bytes = streamer.initialize(in_stream, in_stream_type, rb_lib, 
                                               buffer_size=buffer_size, verbosity=verbosity)
    bytes_processed += n_bytes
    if verbosity > 0: update_progress(float(n_bytes)/in_stream_size)

    # rb_lib should now be fully initialized. Check if files need to be
    # overwritten or if we need to stop to avoid overwriting
    out_files = rb_lib.get_list_of('out_stream')
    for out_file in out_files:
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
    while True:
        chunk_list, n_bytes = streamer.read_chunk(full_only=True, verbosity=verbosity)
        bytes_processed += n_bytes
        if verbosity > 0: update_progress(float(n_bytes)/in_stream_size)
        for rb in chunk_list:
            if rb.loc > n_max: rb.loc = n_max
            n_max -= rb.loc
        write_to_lh5_and_clear(chunk_list, lh5_store)
        if len(chunk_list) == 0 or n_max == 0: break
    if verbosity > 0: update_progress(1)

    # Write out all buffers with any data
    all_rbs = []
    for rb_list in rb_lib.values(): all_rbs += rb_list
    write_to_lh5_and_clear(chunk_list, lh5_store)


    # --------- summary ------------

    elapsed = time.time() - t_start
    print("Time elapsed: {:.2f} sec".format(elapsed))
    if 'sysn' not in raw_file_pattern:
        statinfo = os.stat(raw_file_pattern)
        print('File size: {}'.format(sizeof_fmt(statinfo.st_size)))
        print('Conversion speed: {}ps'.format(sizeof_fmt(statinfo.st_size/elapsed)))
        print('  Output file:', raw_file_pattern)
    else:
        print('Total converted: {}'.format(sizeof_fmt(bytes_processed)))
        print('Conversion speed: {}ps'.format(sizeof_fmt(bytes_processed/elapsed)))

    print('Done.\n')
