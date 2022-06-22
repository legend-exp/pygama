from abc import ABC

from .raw_buffer import RawBuffer, RawBufferLibrary, RawBufferList


class DataStreamer(ABC):
    """ Base clase for data streams

    Provides a uniform interface for streaming, e.g.:

    > header = ds.open_stream(stream_name)
    > for chunk in ds: do_something(chunk)

    Also provides default management of the RawBufferLibrary used for data reading:
    - allocation (if needed)
    - configuration (to match the stream)
    - fill level checking

    Derived classes must define the functions get_decoder_list(), open_stream(),
    and read_packet(); see below.
    """

    def __init__(self):
        self.rb_lib = None
        self.chunk_mode = None
        self.n_bytes_read = 0
        self.any_full = False
        self.packet_id = 0


    def open_stream(self, stream_name, rb_lib=None, buffer_size=8192,
                    chunk_mode='any_full', out_stream='', verbosity=0):
        """ Open and initialize a data stream

        Open the stream, read in the header, set up the buffers

        Call super().initialize([args]) from derived class after loading header
        info to run this default version that sets up buffers in rb_lib using
        the stream's decoders

        Note: this default version has no actual return value! You must overload
        this function, set self.n_bytes_read to the header packet size, and
        return the header data

        Parameters
        ----------
        stream_name : str
            typically a filename or e.g. a port for streaming
        rb_lib : RawBufferLibrary
            A library of buffers for readout from the data stream. rb_lib will
            have its lgdo's initialized during this function
        buffer_size : int
            length of buffers to be read out in read_chunk (for buffers with
            variable length)
        chunk_mode : 'any_full', 'only_full', or 'single_packet'
            sets the mode use for read_chunk
        out_stream : str
            optional name of output stream for default rb_lib generation
        verbosity : int
            verbosity level for the initialize function

        Returns
        -------
        header_data: list(RawBuffer), int
            header_data is a list of RawBuffer's containing any file header
            data, ready for writing to file or further processing. It's not a
            RawBufferList since the buffers may have a different format
        """
        # call super().initialize([args]) to run this default code
        # after loading header info, then follow it with the return call.

        # store chunk mode
        self.chunk_mode = chunk_mode

        # prepare rb_lib -- its lgdo's should still be uninitialized
        if rb_lib is None: rb_lib = self.build_default_rb_lib(out_stream=out_stream)
        self.rb_lib = rb_lib

        # now initialize lgdo's for raw buffers
        decoders = self.get_decoder_list()
        dec_names = []
        for decoder in decoders:
            dec_name = type(decoder).__name__

            # set up wildcard decoder buffers
            if dec_name not in rb_lib:
                if '*' not in rb_lib: continue # user didn't want this decoder
                rb_lib[dec_name] = RawBufferList()
                dec_key = dec_name
                if dec_key.endswith('Decoder'): dec_key = dec_key.removesuffix('Decoder')
                out_name = rb_lib['*'][0].out_name.format(name=dec_key)
                out_stream = rb_lib['*'][0].out_stream.format(name=dec_key)
                key_list = decoder.get_key_list()
                rb = RawBuffer(key_list=key_list, out_stream=out_stream, out_name=out_name)
                rb_lib[dec_name].append(rb)

            # dec_name is in rb_lib: store the name, and initialize its buffer lgdos
            dec_names.append(dec_name)

            # set up wildcard key buffers
            for rb in rb_lib[dec_name]:
                if len(rb.key_list) == 1 and rb.key_list[0] == "*":
                    rb.key_list = decoder.get_key_list()
            keyed_name_rbs = []
            ii = 0
            while ii < len(rb_lib[dec_name]):
                if '{key' in rb_lib[dec_name][ii].out_name:
                    keyed_name_rbs.append(rb_lib[dec_name].pop(ii))
                else: ii += 1
            for rb in keyed_name_rbs:
                for key in rb.key_list:
                    expanded_name = rb.out_name.format(key=key)
                    new_rb = RawBuffer(key_list=[key], out_stream=rb.out_stream, out_name=expanded_name)
                    rb_lib[dec_name].append(new_rb)

            for rb in rb_lib[dec_name]:
                # use the first available key
                key = rb.key_list[0] if len(rb.key_list) > 0 else None
                rb.lgdo = decoder.make_lgdo(key=key, size=buffer_size)

        # make sure there were no entries in rb_lib that weren't among the
        # decoders. If so, just emit a warning and continue.
        if '*' in rb_lib: rb_lib.pop('*')
        for dec_name in rb_lib.keys():
            if dec_name not in dec_names:
                print(f"Warning: no decoder named {dec_name} requested by rb_lib")



    def close_stream(self):
        """ close this data stream. needs to be implemented in derived class """
        pass

    def read_packet(self, verbosity=0):
        """
        Reads a single packet's worth of data in to the rb_lib

        Needs to be overloaded. Gets called by read_chunk()

        Needs to update self.any_full if any buffers would possibly over-fill on
        the next read

        Needs to update self.n_bytes_read too

        Returns
        -------
        still_has_data : bool
            Returns true while there is still data to read
        """
        return True


    def read_chunk(self, chunk_mode_override=None, rp_max=1000000, clear_full_buffers=True, verbosity=0):
        """
        Reads a chunk of data into raw buffers

        Reads packets until at least one buffer is too full to perform another
        read.

        Note: user is responsible for resetting / clearing the raw buffers prior
        to calling read_chunk again.

        Default version just calls read_packet() over and over. Overload as
        necessary.

        Parameters
        ----------
        chunk_mode_override : 'any_full', 'only_full', 'single_packet', or None
            - None : do not override self.chunk_mode
            - 'any_full' : returns all raw buffers with data as soon as any one
              buffer gets full
            - 'only_full' : returns only those raw buffers that became full (or
              nearly full) during the read. This minimizes the number of write calls.
            - 'single_packet' : returns all raw buffers with data after a single
              read is performed. This is useful for streaming data out as soon
              as it is read in (e.g. for diagnostics or in-line analysis)
        rp_max : int
            maximum number of packets to read before returning anyway, even if
            one of the other conditions is not met
        clear_full_buffers : bool
            automatically clear any buffers that report themselves as being full
            prior to reading the chunk. Set to False if clearing manually for a
            minor speed-up
        verbosity : int
            verbosity level for the initialize function

        Returns
        -------
        chunk_list : list of RawBuffers, int
            chunk_list is the list of RawBuffers with data ready for writing to
            file or further processing. The list contains all buffers with data
            or just all full buffers depending on the flag full_only.  Note
            chunk_list is not a RawBufferList since the RawBuffers inside may
            not all have the same structure
        """

        if clear_full_buffers: self.rb_lib.clear_full()
        self.any_full = False

        chunk_mode = self.chunk_mode if chunk_mode_override is None else chunk_mode_override
        if verbosity>1: print(f'reading chunk with chunk_mode {chunk_mode}')

        read_one_packet = (chunk_mode == 'single_packet')
        only_full = (chunk_mode == 'only_full')

        n_packets = 0
        still_has_data = True
        while True:
            still_has_data = self.read_packet(verbosity=verbosity-1)
            if not still_has_data: break
            n_packets += 1
            if read_one_packet or n_packets > rp_max: break
            if self.any_full: break

        # send back all rb's with data if we finished reading
        if not still_has_data: only_full = False

        list_of_rbs = []
        for rb_list in self.rb_lib.values():
            for rb in rb_list:
                if not only_full: # any_full or read_one_packet
                    if rb.loc > 0: list_of_rbs.append(rb)
                elif rb.is_full(): list_of_rbs.append(rb)
        return list_of_rbs




    def get_decoder_list(self):
        """Returns a list of decoder objects for this data stream.
        Needs to be overloaded. Gets called during open_stream().
        """
        return []


    def build_default_rb_lib(self, out_stream=''):
        """ Build the most basic RawBufferLibrary that will work for this stream.

        A RawBufferList containing a single RawBuffer is built for each decoder
        name returned by get_decoder_list. Each buffer's out_name is set to the
        decoder name. The lgdo's do not get initialized.
        """
        rb_lib = RawBufferLibrary()
        decoders = self.get_decoder_list()
        if len(decoders) == 0:
            print(f'No decoders returned by get_decoder_list() for {type(self).__name__}')
            return rb_lib
        for decoder in decoders:
            dec_name = type(decoder).__name__
            dec_key = dec_name
            if dec_key.endswith('Decoder'): dec_key = dec_key.removesuffix('Decoder')
            key_list = decoder.get_key_list()
            rb = RawBuffer(key_list=key_list, out_stream=out_stream, out_name=dec_key)
            rb_lib[dec_name] = RawBufferList()
            rb_lib[dec_name].append(rb)
        return rb_lib
