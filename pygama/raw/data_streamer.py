from abc import ABC

class DataStreamer(ABC):

    def initialize(self, data_source, ch_groups_library=None, buffer_size=8192):
        """
        Returns
        -------
        header_data : list of lgdos
        """
        pass

    def read_chunk(self):
        """
        Returns
        -------
        chunk : list of lgdos
        """
        pass
