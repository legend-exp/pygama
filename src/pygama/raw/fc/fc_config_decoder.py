import logging

from pygama import lgdo

log = logging.getLogger(__name__)

from ..data_decoder import *


class FCConfigDecoder(DataDecoder):
    """
    Decode FlashCam config data

    Derives from DataDecoder in anticipation of possible future functionality;
    currently DataDecoder interface is not used.

    Typical usage:

    fc_config = FCConfigDecoder.decode_config(fcio)

    Then you just use the fcio_config, which is a lgdo Struct (i.e. a dict)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = lgdo.Struct()

    def decode_config(self, fcio):
        config_names = [
            'nsamples', # samples per channel
            'nadcs', # number of adc channels
            'ntriggers', # number of triggertraces
            'telid', # id of telescope
            'adcbits', # bit range of the adc channels
            'sumlength', # length of the fpga integrator
            'blprecision', # precision of the fpga baseline
            'mastercards', # number of attached mastercards
            'triggercards', # number of attached triggercards
            'adccards', # number of attached fadccards
            'gps', # gps mode (0: not used, 1: external pps and 10MHz)
        ]
        for name in config_names:
            if name in self.config:
                log.warning(f'{name} already in self.config. skipping...')
                continue
            value = np.int32(getattr(fcio, name)) # all config fields are int32
            self.config.add_field(name, lgdo.Scalar(value))
        return self.config

    def make_lgdo(self, key=None, size=None): return self.config

    def buffer_is_full(self, rb): return rb.loc > 0
