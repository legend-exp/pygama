from __future__ import annotations

import logging

import fcutils
import numpy as np

from pygama import lgdo
from pygama.raw.data_decoder import DataDecoder
from pygama.raw.raw_buffer import RawBuffer

log = logging.getLogger(__name__)


class FCConfigDecoder(DataDecoder):
    """Decode FlashCam config data.

    Note
    ----
    Derives from :class:`~.raw.data_decoder.DataDecoder` in anticipation of
    possible future functionality. Currently the base class interface is not
    used.

    Example
    -------
    >>> import fcutils
    >>> from pygama.raw.fc.fc_config_decoder import FCConfigDecoder
    >>> fc = fcutils.fcio('file.fcio')
    >>> decoder = FCConfigDecoder()
    >>> config = decoder.decode_config(fc)
    >>> type(config)
    pygama.lgdo.struct.Struct
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = lgdo.Struct()

    def decode_config(self, fcio: fcutils.fcio) -> lgdo.Struct:
        config_names = [
            "nsamples",  # samples per channel
            "nadcs",  # number of adc channels
            "ntriggers",  # number of triggertraces
            "telid",  # id of telescope
            "adcbits",  # bit range of the adc channels
            "sumlength",  # length of the fpga integrator
            "blprecision",  # precision of the fpga baseline
            "mastercards",  # number of attached mastercards
            "triggercards",  # number of attached triggercards
            "adccards",  # number of attached fadccards
            "gps",  # gps mode (0: not used, 1: external pps and 10MHz)
        ]
        for name in config_names:
            if name in self.config:
                log.warning(f"{name} already in self.config. skipping...")
                continue
            value = np.int32(getattr(fcio, name))  # all config fields are int32
            self.config.add_field(name, lgdo.Scalar(value))
        return self.config

    def make_lgdo(self, key: int = None, size: int = None) -> lgdo.Struct:
        return self.config

    def buffer_is_full(self, rb: RawBuffer) -> bool:
        return rb.loc > 0
