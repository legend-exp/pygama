from __future__ import annotations

import json
import logging

import numpy as np

from pygama import lgdo
from pygama.raw.data_decoder import DataDecoder
from pygama.raw.raw_buffer import RawBuffer

log = logging.getLogger(__name__)


class CompassHeaderDecoder(DataDecoder):
    """Decode CoMPASS header data. Also, read in CoMPASS config data if provided.

    JSON Configuration Example
    --------------------------

    Below is an example of the CoMPASS config file; the wf_len and num_enabled_channels are necessary.

    .. code-block:: json

        {
        "model_name": "DT5730",
        "adc_bitcount": 14,
        "sample_rate": 500e6,
        "v_range": 2.0,
        "num_enabled_channels": 1,
        "wf_len": 1000,  # in samples
        }
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = lgdo.Struct()

    def decode_header(self, in_stream: bytes, config_file: str = None) -> lgdo.Struct:
        """Decode the CoMPASS file header, and add CoMPASS config data to the header, if present.

        Parameters
        ----------
        in_stream
            The stream of data to have its header decoded
        config_file
            The config file for the CoMPASS data, if present

        Returns
        -------
        config
            An lgdo.Struct containing the header information, as well as the important config information
            of wf_len and num_enabled_channels
        """
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
            value = np.int32(header_as_list[i])  # all config fields are int32
            self.config.add_field(name, lgdo.Scalar(value))

        # if a json config_file is present, read in the config values to the header
        if isinstance(config_file, str) and config_file.endswith(".json"):
            with open(config_file) as json_file:
                config_file = json.load(json_file)
            # Add the rest of the config file
            for key in config_file:
                self.config.add_field(key, lgdo.Scalar(config_file[key]))
            return self.config

        # if the config_file is None, sacrifice the first packet and write the wf_len to the header,
        # and initialize the number of enabled channels to the max of a CAEN digitizer
        if config_file is None:

            if self.config["energy_short"].value == 1:
                header_length = 25  # if the energy short is present, then there are an extra 2 bytes in the metadata
            else:
                header_length = 23  # the normal packet metadata is 23 bytes long

            # read in the packet metadata
            header_in_bytes = in_stream.read(header_length)

            # get the waveform length so we can read in the rest of the packet
            if header_length == 25:
                [num_samples] = np.frombuffer(header_in_bytes[21:25], dtype=np.uint32)
            if header_length == 23:
                [num_samples] = np.frombuffer(header_in_bytes[19:23], dtype=np.uint32)

            # sacrifice the first waveform so that load_packet() will work correctly, there are 2 bytes per sample
            wf = in_stream.read(2 * num_samples)

            # add the wf_len to the config
            self.config.add_field("wf_len", lgdo.Scalar(num_samples))
            # set the number of enabled channels to the 8 max channels on a CAEN digitizer
            self.config.add_field("num_enabled_channels", lgdo.Scalar(8))

            return self.config

    def make_lgdo(self, key: int = None, size: int = None) -> lgdo.Struct:
        return self.config

    def buffer_is_full(self, rb: RawBuffer) -> bool:
        return rb.loc > 0
