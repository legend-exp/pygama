from abc import ABC

class DataStreamer(ABC):

    def initialize(self, data_source, raw_buffer_library=None, chunk_size=8192, verbosity=0):
        """ Initialize a data stream

        Parameters
        ----------
        data_source : str
            typically a filename or e.g. a port for streaming
        chunk_size : int
            length of buffers to be read out in read_chunk
        verbosity : int
            verbosity level for the initialize function

        Returns
        -------
        header_data, n_bytes: list of RawBuffers, int
            header_data is the returned file header data ready for writing to
            file or further processing (returned as a list for future
            flexibility and to match signature of read_chunk).
            n_bytes is the number of bytes read from the file to extract the
            header
        """
        pass


    def read_chunk(self, full_only=True, verbosity=0):
        """
        Reads a chunk of data into raw buffers

        Note: user is responsible for resetting / clearing the raw buffers prior
        to calling read_chunk again.

        Parameters
        ----------
        full_only : bool
            If true, the returned chunk_list contains only those raw buffers
            that became full during the read. This minimizes the number of
            write calls. 
            If false, returns all raw buffers with data. This is useful for
            streaming data out as soon as it is read in (e.g. for in-line
            analysis)
        verbosity : int
            verbosity level for the initialize function

        Returns
        -------
        chunk_list, n_bytes : list of RawBuffers, int
            chunk_list is the returned list of raw buffers and is ready for
            writing to file or further processing. Note it is not a
            RawBufferList since the RawBuffers inside may not all have the same
            structure
            n_bytes is the number of bytes read from the file during that
            iteration.
        """
        pass
