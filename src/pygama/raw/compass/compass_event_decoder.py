from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np

from pygama import lgdo
from pygama.raw.data_decoder import DataDecoder

log = logging.getLogger(__name__)

compass_decoded_values = {
    # packet index in file
    "packet_id": {
        "dtype": "uint32",
    },
    # ID of board
    "board": {
        "dtype": "uint32",
    },
    # ID of channel recording data
    "channel": {
        "dtype": "uint32",
    },
    # Timestamp of event
    "timestamp": {
        "dtype": "float64",
        "units": "ps",
    },
    # Energy of event
    "energy": {
        "dtype": "uint32",
    },
    # Energy short of event
    "energy_short": {
        "dtype": "uint32",
    },
    # Flags that the digitizer raised
    "flags": {
        "dtype": "uint32",
    },
    # number of samples in a waveform
    "num_samples": {
        "dtype": "int64",
    },
    # waveform data
    "waveform": {
        "dtype": "uint16",
        "datatype": "waveform",
        "wf_len": 65532,  # max value. override this before initializing buffers to save RAM
        "dt": 16,  # override if a different clock rate is used
        "dt_units": "ns",
        "t0_units": "ns",
    },
}


class CompassEventDecoder(DataDecoder):
    """
    Decode CAEN digitizer event data.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.decoded_values = copy.deepcopy(compass_decoded_values)
        super().__init__(*args, **kwargs)
        self.skipped_channels = {}

    def get_key_list(self) -> list[int | str]:
        """
        Get a unique key for each channel present in the CoMPASS file.

        Notes
        -----
        The config file needs the number of channels enabled, because it
        is read into the header and then used here to create the keys.
        If no config file is present, the maximum 8 channels of a
        CAEN digitizer is used, but all events raw_buffers might
        not be filled in that case.
        """
        return range(self.header["num_enabled_channels"].value)

    def set_file_config(self, header: lgdo.Struct) -> None:
        """Access ``header`` members once when each file is opened.
        Parameters
        ----------
        header
            extracted via :meth:`~.compass_header_decoder.CompassHeaderDecoder.decode_header`.
        """
        self.header = header
        self.decoded_values["waveform"]["wf_len"] = self.header["wf_len"].value

    def get_decoded_values(self, channel: int = None) -> dict[str, dict[str, Any]]:
        # CoMPASS uses the same values for all channels
        return self.decoded_values

    def decode_packet(
        self,
        packet: bytes,
        evt_rbkd: lgdo.Table | dict[int, lgdo.Table],
        packet_id: int,
        header: lgdo.Table | dict[int, lgdo.Table],
    ) -> bool:
        """Access ``CoMPASSEvent`` members for each event in the DAQ file.

        Parameters
        ----------
        packet
            The packet to be decoded
        evt_rbkd
            A single table for reading out all data, or a dictionary of tables
            keyed by channel number.
        packet_id
            The index of the packet in the `CoMPASS` stream. Incremented by
            :class:`~.raw.compass.compass_streamer.CompassStreamer`.
        header
            The header of the CoMPASS file, along with user config info,
            that is used to determine waveform lengths and thus buffer sizes

        Returns
        -------
        n_bytes
            (estimated) number of bytes in the packet that was just decoded.
        """
        # Read in the board number and channel number so we can get the right entry in the table
        # Regardless of the length of the header, the first 4 bytes are the board and the channel number
        board = np.frombuffer(packet[0:2], dtype=np.uint16)[0]
        channel = np.frombuffer(packet[2:4], dtype=np.uint16)[0]

        # get the table for this channel
        if channel not in evt_rbkd:
            if channel not in self.skipped_channels:
                self.skipped_channels[channel] = 0
                log.debug(f"Skipping channel: {channel}")
                log.debug(f"evt_rbkd: {evt_rbkd.keys()}")
            self.skipped_channels[channel] += 1
            return False
        tbl = evt_rbkd[channel].lgdo
        ii = evt_rbkd[channel].loc

        # store packet id
        tbl["packet_id"].nda[ii] = packet_id

        # store the info we already have read in
        tbl["board"].nda[ii] = board
        tbl["channel"].nda[ii] = channel

        # the time stamp also does not care about if we have an energy short present
        tbl["timestamp"].nda[ii] = np.frombuffer(packet[4:12], dtype=np.uint64)[0]

        # get the rest of the values depending on if there is an energy_short present
        if header["energy_short"].value == 1:
            tbl["energy"].nda[ii] = np.frombuffer(packet[12:14], dtype=np.uint16)[0]
            tbl["energy_short"].nda[ii] = np.frombuffer(packet[14:16], dtype=np.uint16)[
                0
            ]
            tbl["flags"].nda[ii] = np.frombuffer(packet[16:20], np.uint32)[0]
            tbl["num_samples"].nda[ii] = np.frombuffer(packet[21:25], dtype=np.uint32)[
                0
            ]

            if (
                tbl["num_samples"].nda[ii] != self.decoded_values["waveform"]["wf_len"]
            ):  # make sure that the waveform we read in is the same length as in the config
                raise RuntimeError(
                    f"Waveform size {tbl['num_samples'].nda[ii]} doesn't match expected size {self.decoded_values['waveform']['wf_len']}. "
                    "Skipping packet"
                )

            tbl["waveform"]["values"].nda[ii] = np.frombuffer(
                packet[25:], dtype=np.uint16
            )

        else:
            tbl["energy"].nda[ii] = np.frombuffer(packet[12:14], dtype=np.uint16)[0]
            tbl["energy_short"].nda[ii] = None
            tbl["flags"].nda[ii] = np.frombuffer(packet[14:18], np.uint32)[0]
            tbl["num_samples"].nda[ii] = np.frombuffer(packet[19:23], dtype=np.uint32)[
                0
            ]

            if (
                tbl["num_samples"].nda[ii] != self.decoded_values["waveform"]["wf_len"]
            ):  # make sure that the waveform we read in is the same length as in the config
                raise RuntimeError(
                    f"Waveform size {tbl['num_samples'].nda[ii]} doesn't match expected size {self.decoded_values['waveform']['wf_len']}. "
                    "Skipping packet"
                )

            tbl["waveform"]["values"].nda[ii] = np.frombuffer(
                packet[23:], dtype=np.uint16
            )

        evt_rbkd[channel].loc += 1
        return evt_rbkd[channel].is_full()
