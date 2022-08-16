import logging
import sys
from typing import Any

import numpy as np

from pygama.raw.orca.orca_header import OrcaHeader
from pygama.raw.orca.orca_packet import OrcaPacket
from pygama.raw.raw_buffer import RawBufferLibrary

from .orca_base import OrcaDecoder, get_ccc

log = logging.getLogger(__name__)


"""
Data Format for SIS3316 Digitizer
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


class ORSIS3316WaveformDecoder(OrcaDecoder):
    """Decoder for SIS3316 ADC data written by ORCA."""


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
                "wf_len": 65532,  # max value. override this before initalizing buffers to save RAM
                "dt": 8,  # override if a different clock rate is use
                "dt_units": "ns",
                "t0_units": "ns",
            },
        }

        # self.event_header_length = 1 #?
        self.decoded_values = {}
        self.skipped_channels = {}
        super().__init__(
            header=header, **kwargs
        )  # also initializes the garbage df (whatever that means...)

    def set_header(self, header: OrcaHeader) -> None:
        self.header = header
        import copy

        self.decoded_values = copy.deepcopy(self.decoded_values_template)

        for card_dict in self.header["ObjectInfo"]["Crates"][0]["Cards"]:
            if card_dict["Class Name"] == "ORSIS3316Model":
                card = card_dict["Card"]
                crate = 0
                for channel in range(0, 16):
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

        self.decoded_values["waveform"]["wf_len"] = trace_length


    def get_key_list(self) -> list[int]:
        key_list = []
        for key in self.decoded_values.keys():
            key_list += [key]
        return key_list

    def get_decoded_values(self, key: int = None) -> dict[str, Any]:
        if key is None:
            dec_vals_list = self.decoded_values
            if len(dec_vals_list) == 0:
                raise RuntimeError("decoded_values not built yet!")
                return None
            return dec_vals_list  # Get first thing we find

        if key in self.decoded_values:
            dec_vals_list = self.decoded_values[key]
            return dec_vals_list
        raise RuntimeError("No decoded values for key", key)
        return None

    def decode_packet(
        self, packet: OrcaPacket, packet_id: int, rbl: RawBufferLibrary
    ) -> bool:
        """Decode the ORCA SIS3316 ADC packet."""
        """
        The packet is formatted as
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
         N Data Events follow with format described in manual (NOTE THE FORMAT BITS)
         Also described at the top
        """
        evt_rbkd = rbl.get_keyed_dict()

        evt_data_16 = np.frombuffer(packet, dtype=np.uint16)

        #First grab the crate, card, and channel to calculate ccc
        crate = (packet[1] >> 21) & 0xF
        card = (packet[1] >> 16) & 0x1F
        channel = (packet[1] >> 8) & 0xFF
        ccc = get_ccc(crate, card, channel)

        #Check if this ccc should be recorded.
        if ccc not in evt_rbkd:
            if ccc not in self.skipped_channels:
                self.skipped_channels[ccc] = 0
                log.debug(f"Skipping channel: {ccc}")
                log.debug(f"evt_rbkd: {evt_rbkd.keys()}")
            self.skipped_channels[ccc] += 1
            return False

        orca_header_length = 10
        #Find the number of Events that is in this packet.
        num_of_Events = packet[2]
        num_of_longs = packet[3]
        data_header_length = packet[5]


        #Creating the first table and finding the index.
        tbl = evt_rbkd[ccc].lgdo


        for i in range(0, num_of_Events):
            #will crash if at the end of buffer.
            ii = evt_rbkd[ccc].loc
            if ii <= 8191:
                #save the crate, card, and channel number which does not change
                tbl["crate"].nda[ii] = crate
                tbl["card"].nda[ii] = card
                tbl["channel"].nda[ii] = channel



                """
                calculates where to start indexing based on the num_of_longs
                for each event offset by the orca_header_length.
                """
                event_start = orca_header_length + num_of_longs*i

                if len(packet) > 10:
                    tbl["timestamp"].nda[ii] = packet[event_start + 1] + ((packet[event_start] & 0xFFFF0000) << 16)
                else:
                    tbl["timestamp"].nda[ii] = 0

                information = (packet[event_start+3] & 0xFF000000)


                data_header_length16 = data_header_length*2


                expected_wf_length = num_of_longs*2 - data_header_length16


                i_wf_start = data_header_length16 + event_start*2

                i_wf_stop = i_wf_start + expected_wf_length


                if expected_wf_length > 0:
                    tbl["waveform"]["values"].nda[ii] = evt_data_16[i_wf_start:i_wf_stop]

                #move to the next index for the next event.
                evt_rbkd[ccc].loc += 1

        return evt_rbkd[ccc].is_full()
