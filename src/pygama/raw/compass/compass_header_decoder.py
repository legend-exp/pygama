from __future__ import annotations

import logging

from pygama import lgdo
from pygama.raw.compass.compass_config_parser import compass_config_to_struct
from pygama.raw.data_decoder import DataDecoder
from pygama.raw.raw_buffer import RawBuffer

log = logging.getLogger(__name__)


class CompassHeaderDecoder(DataDecoder):
    """
    Decode CoMPASS header data. Also, read in CoMPASS config data if provided using the compass_config_parser
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = None  # initialize to none, because compass_config_to_struct always returns a struct

    def decode_header(
        self, in_stream: bytes, config_file: str = None, wf_len: int = None
    ) -> dict:
        """Decode the CoMPASS file header, and add CoMPASS config data to the header, if present.

        Parameters
        ----------
        in_stream
            The stream of data to have its header decoded
        config_file
            The config file for the CoMPASS data, if present
        wf_len
            The length of the first waveform in the file, only pre-calculated when the config_file is none

        Returns
        -------
        config
            A dict containing the header information, as well as the important config information
            of wf_len and num_enabled_channels
        """
        self.config = compass_config_to_struct(config_file, wf_len)

        config_names = [
            "energy_channels",  # energy is given in channels (0: false, 1: true)
            "energy_calibrated",  # energy is given in keV/MeV, according to the calibration (0: false, 1: true)
            "energy_short",  # energy short is present (0: false, 1: true)
            "waveform_samples",  # waveform samples are present (0: false, 1: true)
        ]
        header_in_bytes = in_stream.read(
            2
        )  # we always have to read in the first 2 bytes of the header for CoMPASS v2 files
        header_in_binary = bin(int.from_bytes(header_in_bytes, byteorder="little"))
        header_as_list = str(header_in_binary)[
            ::-1
        ]  # reverse it as we care about bit 0, bit 1, etc.

        for i, name in enumerate(config_names):
            if name in self.config:
                log.warning(f"{name} already in self.config. skipping...")
                continue
            value = int(header_as_list[i])
            self.config.add_field(
                str(name), lgdo.Scalar(value)
            )  # self.config is a struct

        return self.config

    def make_lgdo(self, key: int = None, size: int = None) -> lgdo.Struct:
        if self.config is None:
            raise RuntimeError(
                "self.config still None, need to decode header before calling make_lgdo"
            )
        return self.config  # self.config is already an lgdo, namely it is a struct

    def buffer_is_full(self, rb: RawBuffer) -> bool:
        return rb.loc > 0
