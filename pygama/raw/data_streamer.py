from abc import ABC

class DataStreamer(ABC):

    def initialize(self, data_source, ch_groups_library=None, chunk_size=8192, verbosity=0):
        """ Initialize a data stream

        Parameters
        ----------
        data_source : str
            typically a filename or e.g. a port for streaming
        chunk_size : int
            length of tables to be read out in read_chunk
        verbosity : int
            verbosity level for the initialize function

        Returns
        -------
        header_data : list of lgdos
            the returned file header data is ready for writing to file or
            further processing
        """
        pass

    def read_chunk(self, verbosity=0):
        """
        Reads until any lgdo is ready for writing / further processing, and
        returns all ready lgdo's in a list

        Note: user is responsible for resetting / clearing the LGDOs prior to
        calling read_chunk again.

        Parameters
        ----------
        verbosity : int
            verbosity level for the initialize function

        Returns
        -------
        chunk_list, n_bytes : ntuple (list of lgdos, int)
            chunk_list is the returned list of lgdos (tables, etc) and is ready
            for writing to file or further processing.
            n_bytes is the number of bytes read from the file during that
            iteration.
        """
        pass
