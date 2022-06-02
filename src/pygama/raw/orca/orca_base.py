from ..data_decoder import *


class OrcaDecoder(DataDecoder):
    """ Base class for ORCA decoders.

    Mostly here to provide a standard interface for setting the header in init
    """
    def __init__(self, header=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.header = None
        if header: self.set_header(header) # allow for derived class function to be called


    def set_header(self, header):
        ''' Setter for headers. Overload to set card parameters, etc. '''
        self.header = header


# define a standard hash for crate, card, channel <--> integer
def get_ccc(crate, card, channel):
    return (crate << 9) + ((card & 0x1f) << 4) + (channel & 0xf)


def get_crate(ccc):
    return ccc >> 9


def get_card(ccc):
    return (ccc >> 4) & 0x1f


def get_channel(ccc):
    return ccc & 0xf
