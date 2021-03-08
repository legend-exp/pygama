from abc import ABC

class DataStreamer(ABC):

    def initialize(self, data_source, ch_groups_library=None, buffer_size=8192, verbosity=0):
        """
        Returns
        -------
        header_data : list of lgdos
            the returned header_data is ready for writing to file or further
            processing
        """
        pass

    def read_chunk(self, verbosity=0):
        """
        Reads until any lgdo is ready for writing / further processing, and
        returns all ready lgdo's in a list

        Note: user is responsible for resetting / clearing the LGDOs prior to
        calling read_chunk again.

        Returns
        -------
        chunk, n_bytes : ntuple ( list of lgdos, int)
            chunk is the returned list of lgdos (tables, etc) is ready for writing to
            file or further processing
            n_bytes is the number of bytes read from the file during that
            iteration.
        """
        pass
