from abc import ABC
from raw_buffer import RawBufferLibrary

class DataStreamer(ABC):
    """ Base clase for data streams """

    def __init__(self)
        self.rb_lib = None

    def initialize(self, in_stream, rb_lib, buffer_size=8192, verbosity=0):
        """ Initialize a data stream

        Open the in_stream, read in the header, set up the buffers

        Parameters
        ----------
        in_stream : str
            typically a filename or e.g. a port for streaming
        rb_lib : RawBufferLibrary
            A library of buffers for readout from the data stream. rb_lib should
            be initialized during this function
        buffer_size : int
            length of buffers to be read out in read_chunk
        verbosity : int
            verbosity level for the initialize function

        Returns
        -------
        header_data, n_bytes: list(RawBuffer), int
            header_data is a list of RawBuffer's containing any file header
                data, ready for writing to file or further processing. It's not
                a RawBufferList since the buffers may have a different format
            n_bytes is the number of bytes read from the file to extract the
                header data
        """
        # call super().initialize(*args, **kwargs) to run this default code
        # after loading header info, then follow it with the return call.
        self.rb_lib = rb_lib
        decoders = self.get_decoder_list()
        dec_names = []
        for decoder in decoders:
            dec_name = type(decoder).__name__

            # set up wildcard buffers
            if dec_name not in rb_lib 
                if '*' not in rb_lib: continue # user didn't want this decoder
                dec_key = dec_name
                if dec_key.endswith('Decoder'): dec_key.removesuffix('Decoder')
                out_name = rb_lib['*'][0].out_name.format('name'=dec_key)
                out_stream = rb_lib['*'][0].out_stream.format('name'=dec_key)
                rb = RawBuffer(out_stream=out_stream, out_name=decoder)
                rb_lib[dec_name] = RawBufferList()
                rb_lib[dec_name].append(rb)

            # dec_name is in rb_lib: store the name, and initialize its buffer lgdos
            dec_names.append(dec_name)
            rb_lib[dec_name].make_lgdos(decoder, size=buffer_size)

        # make sure there were no entries in rb_lib that weren't among the
        # decoders. If so, just emit a warning and continue.
        for dec_name in rb_lib.keys():
            if dec_name not in dec_names:
                print(f"Warning: no decoder named {dec_name} requested by rb_lib")



    def read_chunk(self, full_only=True, verbosity=0):
        """
        Reads a chunk of data into raw buffers

        Note: user is responsible for resetting / clearing the raw buffers prior
        to calling read_chunk again.

        Parameters
        ----------
        full_only : bool
            If true, the returned chunk_list contains only those raw buffers
            that became full (or nearly full) during the read. This minimizes
            the number of write calls. 
            If false, returns all raw buffers with data. This is useful for
            streaming data out as soon as it is read in (e.g. for in-line
            analysis)
        verbosity : int
            verbosity level for the initialize function

        Returns
        -------
        chunk_list, n_bytes : list of RawBuffers, int
            chunk_list is the list of RawBuffers with data ready for writing to
                file or further processing. The list contains all buffers with
                data or just all full buffers depending on the flag full_only.
                Note chunk_list is not a RawBufferList since the RawBuffers
                inside may not all have the same structure
            n_bytes is the number of bytes read from the file during this
                iteration.
        """
        
        pass


    def get_decoder_list(self):
        """Returns a list of decoder objects for this data stream.

        Needs to be overloaded. Should be called after the stream is opened /
        initialized.
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
            rb = RawBuffer(out_stream=out_stream, out_name=decoder)
            rb_lib[decoder] = RawBufferList()
            rb_lib[decoder].append(rb)

