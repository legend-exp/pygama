import copy
import logging
import sys
from typing import Any

import numpy as np

from pygama.raw.orca.orca_base import OrcaDecoder, get_ccc
from pygama.raw.orca.orca_header import OrcaHeader
from pygama.raw.orca.orca_packet import OrcaPacket
from pygama.raw.raw_buffer import RawBufferLibrary

log = logging.getLogger(__name__)


class ORSIS3302DecoderForEnergy(OrcaDecoder):
    """Decoder for `Struck SIS3302 <https://www.struck.de/sis3302.htm>`_ digitizer
    data written by ORCA.
    """

    def __init__(self, header: OrcaHeader = None, **kwargs) -> None:

        self.decoded_values_template = {
            "packet_id": {
                "dtype": "uint32",
            },
            "energy": {
                "dtype": "uint32",
                "units": "adc",
            },
            "energy_first": {
                "dtype": "uint32",
            },
            "timestamp": {
                "dtype": "uint64",
                "units": "clock_ticks",
            },
            "crate": {
                "dtype": "uint8",
            },
            "card": {
                "dtype": "uint8",
            },
            "channel": {
                "dtype": "uint8",
            },
            "waveform": {
                "dtype": "uint16",
                "datatype": "waveform",
                "wf_len": 65532,  # max value. override this before initializing buffers to save RAM
                "dt": 10,  # override if a different clock rate is used
                "dt_units": "ns",
                "t0_units": "ns",
            },
        }
        self.decoded_values = {}
        super().__init__(header=header, **kwargs)
        self.skipped_channels = {}

    def set_header(self, header: OrcaHeader) -> None:
        self.header = header

        obj_info_dict = header.get_object_info("ORSIS3302Model")
        # Loop over crates, cards, build decoded values for enabled channels
        for crate in obj_info_dict:
            for card in obj_info_dict[crate]:
                int_enabled_mask = obj_info_dict[crate][card][
                    "internalTriggerEnabledMask"
                ]
                ext_enabled_mask = obj_info_dict[crate][card][
                    "externalTriggerEnabledMask"
                ]
                enabled_mask = int_enabled_mask | ext_enabled_mask
                for channel in range(8):
                    # only care about enabled channels
                    if not ((enabled_mask >> channel) & 0x1):
                        continue

                    ccc = get_ccc(crate, card, channel)

                    self.decoded_values[ccc] = copy.deepcopy(
                        self.decoded_values_template
                    )

                    # get trace length(s). Should all be the same until
                    # multi-buffer mode is implemented AND each channel has its
                    # own buffer
                    trace_length = obj_info_dict[crate][card]["sampleLengths"][
                        int(channel / 2)
                    ]
                    if trace_length <= 0 or trace_length > 2**16:
                        raise RuntimeError(f"invalid trace_length {trace_length}")
                        sys.exit()
                    self.decoded_values[ccc]["waveform"]["wf_len"] = trace_length

    def get_key_lists(self) -> list[str]:
        key_lists = []
        for key in self.decoded_values.keys():
            key_lists.append([key])
        return key_lists

    def get_decoded_values(self, key: int = None) -> dict[str, Any]:
        if key is None:
            dec_vals_list = self.decoded_values.values()
            if len(dec_vals_list) >= 0:
                return list(dec_vals_list)[0]
            raise RuntimeError("decoded_values not built")
        if key in self.decoded_values:
            return self.decoded_values[key]
        raise KeyError(f"no decoded values for key {key}")

    def decode_packet(
        self, packet: OrcaPacket, packet_id: int, rbl: RawBufferLibrary
    ) -> bool:
        evt_rbkd = rbl.get_keyed_dict()

        # read the crate/card/channel first
        crate = (packet[1] >> 21) & 0xF
        card = (packet[1] >> 16) & 0x1F
        channel = (packet[1] >> 8) & 0xFF
        ccc = get_ccc(crate, card, channel)

        # get the table for this crate/card/channel
        if ccc not in evt_rbkd:
            if ccc not in self.skipped_channels:
                self.skipped_channels[ccc] = 0
                log.debug(f"Skipping channel: {ccc}")
                log.debug(f"evt_rbkd: {evt_rbkd.keys()}")
            self.skipped_channels[ccc] += 1
            return False
        tbl = evt_rbkd[ccc].lgdo
        ii = evt_rbkd[ccc].loc

        # store packet id
        tbl["packet_id"].nda[ii] = packet_id

        # read the rest of the record
        tbl["crate"].nda[ii] = crate
        tbl["card"].nda[ii] = card
        tbl["channel"].nda[ii] = channel
        buffer_wrap = packet[1] & 0x1
        wf_length32 = packet[2]
        ene_wf_length32 = packet[3]
        tbl["timestamp"].nda[ii] = packet[5] + (
            (packet[4] & 0xFFFF0000) << 16
        )  # might need to convert to uint64
        last_word = packet[-1]

        # get the footer
        tbl["energy"].nda[ii] = packet[-4]
        tbl["energy_first"].nda[ii] = packet[-3]

        # Interpret the raw event data into numpy array of 16 bit ints.
        # Does not copy data. p16 is read-only
        p16 = np.frombuffer(packet, dtype=np.uint16)

        # compute expected and actual array dimensions
        wf_length16 = 2 * wf_length32
        ene_wf_length16 = 2 * ene_wf_length32
        orca_header_length16 = 8
        sis_header_length16 = 8 if buffer_wrap else 4
        header_length16 = orca_header_length16 + sis_header_length16
        footer_length16 = 8
        expected_wf_length16 = (
            len(p16) - header_length16 - footer_length16 - ene_wf_length16
        )

        # error check: waveform size must match expectations
        if wf_length16 != expected_wf_length16 or last_word != 0xDEADBEEF:
            raise RuntimeError(
                f"Waveform size {wf_length16} doesn't match expected size {expected_wf_length16}. "
                "The Last Word (should be 0xdeadbeef): {hex(last_word)}"
            )

        # splitting waveform indices into two chunks (all referring to the 16 bit array)
        i_wf_start = header_length16
        i_wf_stop = i_wf_start + wf_length16
        if buffer_wrap:
            # start somewhere in the middle of the record
            i_start_1 = packet[7] + header_length16 + 1
            i_stop_1 = i_wf_stop  # end of the wf record
            i_start_2 = i_wf_start  # beginning of the wf record
            i_stop_2 = i_start_1

        # handle the waveform(s)
        tbwf = tbl["waveform"]["values"].nda[ii]
        if wf_length32 > 0:
            if not buffer_wrap:
                if i_wf_stop - i_wf_start != expected_wf_length16:
                    raise RuntimeError(
                        f"We expected {expected_wf_length16} waveform samples "
                        "and only got {i_wf_stop-i_wf_start}"
                    )
                tbwf[:expected_wf_length16] = p16[i_wf_start:i_wf_stop]
            else:
                len1 = i_stop_1 - i_start_1
                len2 = i_stop_2 - i_start_2
                if len1 + len2 != expected_wf_length16:
                    raise RuntimeError(
                        f"We expected {expected_wf_length16} waveform samples and only got {len1+len2}"
                    )
                tbwf[:len1] = p16[i_start_1:i_stop_1]
                tbwf[len1 : len1 + len2] = p16[i_start_2:i_stop_2]

        evt_rbkd[ccc].loc += 1
        return evt_rbkd[ccc].is_full()


class ORSIS3316WaveformDecoder(OrcaDecoder):
    """Decoder for `Struck SIS3316 <https://www.struck.de/sis3316.html>`_
    digitizer data written by ORCA."""

    def __init__(self, header: OrcaHeader = None, **kwargs) -> None:

        # store an entry for every event
        self.decoded_values_template = {
            "packet_id": {
                "dtype": "uint32",
            },
            "energy": {
                "dtype": "uint32",
                "units": "adc",
            },
            "peakHighValue": {
                "dtype": "uint32",
                "units": "adc",
            },
            "peakHighIndex": {
                "dtype": "uint32",
                "units": "adc",
            },
            "accSum1": {
                "dtype": "uint32",
                "units": "adc",
            },
            "information": {
                "dtype": "uint32",
            },
            "accSum2": {
                "dtype": "uint32",
                "units": "adc",
            },
            "accSum3": {
                "dtype": "uint32",
                "units": "adc",
            },
            "accSum4": {
                "dtype": "uint32",
                "units": "adc",
            },
            "accSum5": {
                "dtype": "uint32",
                "units": "adc",
            },
            "accSum6": {
                "dtype": "uint32",
                "units": "adc",
            },
            "accSum7": {
                "dtype": "uint32",
                "units": "adc",
            },
            "accSum8": {
                "dtype": "uint32",
                "units": "adc",
            },
            "mawMax": {
                "dtype": "uint32",
                "units": "adc",
            },
            "mawBefore": {
                "dtype": "uint32",
                "units": "adc",
            },
            "mawAfter": {
                "dtype": "uint32",
                "units": "adc",
            },
            "startEnergy": {
                "dtype": "uint32",
                "units": "adc",
            },
            "maxEnergy": {
                "dtype": "uint32",
                "units": "adc",
            },
            "energy_first": {
                "dtype": "uint32",
            },
            "timestamp": {
                "dtype": "uint64",
                "units": "clock_ticks",
            },
            "crate": {
                "dtype": "uint8",
            },
            "card": {
                "dtype": "uint8",
            },
            "channel": {
                "dtype": "uint8",
            },
            # waveform data
            "waveform": {
                "dtype": "uint16",
                "datatype": "waveform",
                "wf_len": 65532,  # max value. override this before initializing buffers to save RAM
                "dt": 8,  # override if a different clock rate is used
                "dt_units": "ns",
                "t0_units": "ns",
            },
        }

        self.decoded_values = {}
        self.skipped_channels = {}
        super().__init__(
            header=header, **kwargs
        )  # also initializes the garbage df (whatever that means...)

    def set_header(self, header: OrcaHeader) -> None:
        self.header = header

        for card_dict in self.header["ObjectInfo"]["Crates"][0]["Cards"]:
            if card_dict["Class Name"] == "ORSIS3316Model":
                channel_list = [int(d) for d in str(bin(card_dict["enabledMask"]))[2:]]
                card = card_dict["Card"]
                crate = 0
                for i in range(0, len(channel_list)):
                    if int(channel_list[i]) == 1:
                        channel = i
                        ccc = get_ccc(crate, card, channel)
                        trace_length = card_dict["rawDataBufferLen"]
                        self.decoded_values[ccc] = copy.deepcopy(
                            self.decoded_values_template
                        )

                        if trace_length <= 0 or trace_length > 2**16:
                            raise RuntimeError(
                                "invalid trace_length: ",
                                trace_length,
                            )

                        self.decoded_values[ccc]["waveform"]["wf_len"] = trace_length
                    else:
                        continue

    def get_key_lists(self) -> list[int]:
        key_lists = []
        for key in self.decoded_values.keys():
            key_lists.append([key])
        return key_lists

    def get_decoded_values(self, key: int = None) -> dict[str, Any]:
        if key is None:
            dec_vals_list = self.decoded_values.values()
            if len(dec_vals_list) == 0:
                raise RuntimeError("decoded_values not built yet!")

            return dec_vals_list  # Get first thing we find
        else:
            dec_vals_list = self.decoded_values[key]
            return dec_vals_list
        raise RuntimeError("no decoded values for key", key)

    def decode_packet(
        self, packet: OrcaPacket, packet_id: int, rbl: RawBufferLibrary
    ) -> bool:
        """Decode the ORCA SIS3316 ADC packet.

        The packet is formatted as:

        .. code-block:: text

            xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx
            ^^^^ ^^^^ ^^^^ ^^----------------------- Data ID (from header)
            -----------------^^ ^^^^ ^^^^ ^^^^ ^^^^- length
            xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx
            --------^-^^^--------------------------- Crate number
            -------------^-^^^^--------------------- Card number
            --------------------^^^^ ^^^^----------- Chan number
            xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  Num Events in this packet
            xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  Num longs in each record
            xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  Num of Records that were in the FIFO
            xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  Num of longs in data header -- can get from the raw data also
            xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  Spare
            xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  Spare
            xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  Spare
            xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  Spare

        Then, data events follow with format described in below (note the format bits):

        .. code-block:: text

            xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx
            -----------------------------------^^^^- Format bits (from header)
            --------------------^^^^ ^^^^ ^^^^------ Channel ID
            ^^^^ ^^^^ ^^^^ ^^^^--------------------- Timestamp[47:32]
            xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  Timestamp[31:0]
            if Format bit 0 = 1 add
            xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx
            --------------------^^^^ ^^^^ ^^^^ ^^^^- Peakhigh value
            ^^^^ ^^^^ ^^^^ ^^^^--------------------- Index of Peakhigh value
            xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx
            ----------^^^^ ^^^^ ^^^^ ^^^^ ^^^^ ^^^^- Accumulator sum of Gate 1 [23:0]
            ^^^^ ^^^^------------------------------- Information byte
            xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx Accumulator sum of Gate 2 [27:0]
            xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx Accumulator sum of Gate 3 [27:0]
            xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx Accumulator sum of Gate 4 [27:0]
            xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx Accumulator sum of Gate 5 [27:0]
            xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx Accumulator sum of Gate 6 [27:0]
            If Format bit 1 = 1 add
            xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx Accumulator sum of Gate 7 [27:0]
            xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx Accumulator sum of Gate 8 [27:0]
            If Format bit 2 = 1 add
            xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx MAW Maximum value [27:0]
            xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx MAW Value before Trigger [27:0]
            xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx MAW Value after/with Trigger [27:0]
            If Format bit 3 = 1 add
            xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx Start Energy Value (in Trigger Gate)
            xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx Max Energy Value (in Trigger Gate)
            Regardless of format bit
            xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx
            -------^^ ^^^^ ^^^^ ^^^^ ^^^^ ^^^^ ^^^^- number of raw samples divided by 2
            ------^--------------------------------- status Flag
            -----^---------------------------------- MAW Test Flag
            Followed by N ADC Raw Samples (2 Samples per 32 bit word)
            Followed by MAW Test data
        """

        evt_rbkd = rbl.get_keyed_dict()

        evt_data_16 = np.frombuffer(packet, dtype=np.uint16)

        # First grab the crate, card, and channel to calculate ccc
        crate = (packet[1] >> 21) & 0xF
        card = (packet[1] >> 16) & 0x1F
        channel = (packet[1] >> 8) & 0xFF
        ccc = get_ccc(crate, card, channel)

        # Check if this ccc should be recorded.
        if ccc not in evt_rbkd:
            if ccc not in self.skipped_channels:
                self.skipped_channels[ccc] = 0
                log.debug(f"Skipping channel: {ccc}")
                log.debug(f"evt_rbkd: {evt_rbkd.keys()}")
            self.skipped_channels[ccc] += 1
            return False

        orca_header_length = 10
        # Find the number of Events that is in this packet.
        num_of_events = packet[2]
        num_of_longs = packet[3]
        data_header_length = packet[5]

        # Creating the first table and finding the index.
        tbl = evt_rbkd[ccc].lgdo

        for i in range(0, num_of_events):
            # will crash if at the end of buffer.
            ii = evt_rbkd[ccc].loc
            if ii <= 8191:
                # save the crate, card, and channel number which does not change
                tbl["crate"].nda[ii] = crate
                tbl["card"].nda[ii] = card
                tbl["channel"].nda[ii] = channel

                # calculates where to start indexing based on the num_of_longs
                # for each event offset by the orca_header_length.
                event_start = orca_header_length + num_of_longs * i

                if len(packet) > 10:
                    tbl["timestamp"].nda[ii] = packet[event_start + 1] + (
                        (packet[event_start] & 0xFFFF0000) << 16
                    )
                else:
                    tbl["timestamp"].nda[ii] = 0

                index = event_start + 2
                if (packet[event_start] & 0x1) == 1:
                    tbl["peakHighValue"].nda[ii] = packet[index] & 0xFFFF
                    tbl["peakHighIndex"].nda[ii] = (packet[index] >> 16) & 0xFFFF
                    index += 1
                    tbl["accSum1"].nda[ii] = packet[index] & 0xFFFFFF
                    tbl["information"].nda[ii] = (packet[index] >> 24) & 0xFF
                    index += 1
                    tbl["accSum2"].nda[ii] = packet[index]
                    index += 1
                    tbl["accSum3"].nda[ii] = packet[index]
                    index += 1
                    tbl["accSum4"].nda[ii] = packet[index]
                    index += 1
                    tbl["accSum5"].nda[ii] = packet[index]
                    index += 1
                    tbl["accSum6"].nda[ii] = packet[index]
                else:
                    tbl["peakHighValue"].nda[ii] = -1
                    tbl["peakHighIndex"].nda[ii] = -1
                    tbl["accSum1"].nda[ii] = -1
                    tbl["accSum2"].nda[ii] = -1
                    tbl["accSum3"].nda[ii]
                    tbl["accSum4"].nda[ii]
                    tbl["accSum5"].nda[ii] = -1
                    tbl["accSum6"].nda[ii] = -1

                if ((packet[event_start] >> 1) & 0x1) == 1:
                    tbl["accSum7"].nda[ii] = packet[index]
                    index += 1
                    tbl["accSum8"].nda[ii] = packet[index]
                    index += 1
                else:
                    tbl["accSum7"].nda[ii] = -1
                    tbl["accSum8"].nda[ii] = -1

                if ((packet[event_start] >> 2) & 0x1) == 1:
                    tbl["mawMax"].nda[ii] = packet[index]
                    index += 1
                    tbl["mawBefore"].nda[ii] = packet[index]
                    index += 1
                    tbl["mawAfter"].nda[ii] = packet[index]
                    index += 1
                else:
                    tbl["mawMax"].nda[ii] = -1
                    tbl["mawBefore"].nda[ii] = -1
                    tbl["mawAfter"].nda[ii] = -1

                if ((packet[event_start] >> 3) & 0x1) == 1:
                    tbl["startEnergy"].nda[ii] = packet[index]
                    index += 1
                    tbl["maxEnergy"].nda[ii] = packet[index]
                    index += 1
                else:
                    tbl["startEnergy"].nda[ii] = -1
                    tbl["maxEnergy"].nda[ii] = -1

                data_header_length16 = data_header_length * 2

                expected_wf_length = num_of_longs * 2 - data_header_length16

                i_wf_start = data_header_length16 + event_start * 2

                i_wf_stop = i_wf_start + expected_wf_length

                if expected_wf_length > 0:
                    tbl["waveform"]["values"].nda[ii] = evt_data_16[
                        i_wf_start:i_wf_stop
                    ]

                # move to the next index for the next event.
                evt_rbkd[ccc].loc += 1

        return evt_rbkd[ccc].is_full()
