from __future__ import annotations

import logging

import numpy as np

from pygama.raw.compass.compass_event_decoder import CompassEventDecoder
from pygama.raw.compass.compass_header_decoder import CompassHeaderDecoder
from pygama.raw.data_decoder import DataDecoder
from pygama.raw.data_streamer import DataStreamer
from pygama.raw.raw_buffer import RawBuffer, RawBufferLibrary

log = logging.getLogger(__name__)


class CompassStreamer(DataStreamer):
    """Data streamer for CoMPASS data streams."""

    def __init__(self, config_file) -> None:
        super().__init__()
        self.in_stream = None
        self.header = None
        self.buffer = np.empty(
            1024, dtype="bytes"
        )  # create a buffer that is around 4 kB
        self.header_decoder = CompassHeaderDecoder()
        self.event_decoder = CompassEventDecoder()
        self.config_file = (
            config_file  # optional config file to be passed when calling build_raw
        )
        self.event_rbkd = None

    def get_decoder_list(self) -> list[DataDecoder]:
        dec_list = []
        dec_list.append(self.header_decoder)
        dec_list.append(self.event_decoder)
        return dec_list

    def open_stream(
        self,
        stream_name: str,
        rb_lib: RawBufferLibrary = None,
        buffer_size: int = 8192,
        chunk_mode: str = "any_full",
        out_stream: str = "",
    ) -> tuple[list[RawBuffer], int]:
        """Initialize the CoMPASS data stream. If a config file is present, load just the 2 byte header.
        If a config file is absent, read the first packet to determine waveform length, and initialize the
        keys to the max number of channels of a CAEN digitizer.

        Parameters
        ----------
        stream_name
            The CoMPASS filename. Only file streams are currently supported.
            Socket stream reading can be added later.
        rb_lib
            library of buffers for this stream.
        buffer_size
            length of tables to be read out in :meth:`read_chunk`.
        chunk_mode : 'any_full', 'only_full', or 'single_packet'
            sets the mode use for :meth:`read_chunk`.
        out_stream
            optional name of output stream for default `rb_lib` generation.

        Returns
        -------
        header_data
            a list of length 1 containing the raw buffer holding the CoMPASS header.

        Notes
        -----
        CoMPASS files have a header that is only 2 bytes long and only appears at the top of the file.
        So, we must read this header once, and then proceed to read packets in.
        """
        # set the in_stream
        self.set_in_stream(stream_name)
        self.n_bytes_read = 0

        # read in and decode the file header info if the config file is present
        if self.config_file is not None:
            self.header = self.header_decoder.decode_header(
                self.in_stream, self.config_file
            )  # returns an lgdo.Struct
            self.n_bytes_read += (
                2  # there are 2 bytes in the header, for a 16 bit number to read out
            )

            # set up data loop variables
            self.packet_id = 0

        # read in the header and get the wf_len by sacrificing the first packet if the config_file is not present
        # also set the keys to the maximum number of channels available to a CAEN digitizer
        else:
            self.header = self.header_decoder.decode_header(
                self.in_stream, self.config_file
            )  # returns an lgdo.Struct

            # set up data loop variables
            self.packet_id = 1

            # the number of bytes read is different if the energy_short is present in the header
            if self.header["energy_short"].value == 1:
                self.n_bytes_read += (
                    2 + 25 + 2 * self.header["wf_len"].value
                )  # there are 2 bytes in the header, 25 in packet metadata, and 2*num_samples in bytes
            else:
                self.n_bytes_read += 2 + 23 + 2 * self.header["wf_len"].value

        # The wf_len from the config_file or from the first packet is then used to initialize raw_buffers to the correct size
        self.event_decoder.set_file_config(self.header)

        # initialize the buffers in rb_lib. Store them for fast lookup
        super().open_stream(
            stream_name,
            rb_lib,
            buffer_size=buffer_size,
            chunk_mode=chunk_mode,
            out_stream=out_stream,
        )
        if rb_lib is None:
            rb_lib = self.rb_lib

        # set up the event data raw buffers
        self.event_rbkd = (
            rb_lib["CompassEventDecoder"].get_keyed_dict()
            if "CompassEventDecoder" in rb_lib
            else None
        )

        if "CompassHeaderDecoder" in rb_lib:
            header_rb_list = rb_lib["CompassHeaderDecoder"]
            if len(header_rb_list) != 1:
                log.warning(
                    f"header_rb_list had length {len(header_rb_list)}, ignoring all but the first"
                )
            rb = header_rb_list[0]
        else:
            rb = RawBuffer(lgdo=self.header)
        rb.loc = 1  # we have filled this buffer
        return [rb]

    def set_in_stream(self, stream_name: str) -> None:
        """Sets the in_stream by opening a binary file"""
        if self.in_stream is not None:
            self.in_stream.close()
            self.in_stream = None
        else:
            self.in_stream = open(
                stream_name.encode("utf-8"), "rb"
            )  # CoMPASS files are binary
        self.n_bytes_read = 0

    def close_stream(self) -> None:
        """Close this data stream."""
        if self.in_stream is None:
            raise RuntimeError("tried to close an unopened stream")
        self.in_stream.close()
        self.in_stream = None

    # TODO: add a config file that allows users to specify if "ADC Channels" or "Calibrated" or "Both" options were enabled, as well as Vpp
    def load_packet(self) -> np.uint32 | None:
        """Loads the next packet into the internal buffer.

        Returns packet as a :class:`numpy.uint32` view of the buffer (a slice),
        returns ``None`` at EOF.

        Notes
        -----
        First, load_packet finds the packet's metadata length in bytes by looking at the header,
        and then reads the metadata into the buffer. From the metadata, the length of the waveform can be determined and then the buffer is
        resized to fit the metadata and the waveform. Finally, the buffer is filled with the packet and then returned.

        """
        if self.in_stream is None:
            raise RuntimeError("self.in_stream is None")

        if (self.packet_id == 0) and (self.n_bytes_read != 2):
            raise RuntimeError(
                f"The 2 byte filer header was not converted, instead read in {self.n_bytes_read} for the file header"
            )

        # packets have metadata of variable lengths, depending on if the header shows that energy_short is present in the metadata
        if self.header["energy_short"].value == 1:
            header_length = 25  # if the energy short is present, then there are an extra 2 bytes in the metadata
        else:
            header_length = 23  # the normal packet metadata is 23 bytes long

        # read the packet's metadata into the buffer
        pkt_hdr = self.buffer[:header_length]
        n_bytes_read = self.in_stream.readinto(pkt_hdr)
        self.n_bytes_read += n_bytes_read

        # return None once we run out of file
        if n_bytes_read == 0:
            return None
        if (n_bytes_read != 25) and (n_bytes_read != 23):
            raise RuntimeError(f"only got {n_bytes_read} bytes for packet header")

        # get the waveform length so we can read in the rest of the packet
        if n_bytes_read == 25:
            [num_samples] = np.frombuffer(pkt_hdr[21:25], dtype=np.uint32)
            pkt_length = header_length + 2 * num_samples
        if n_bytes_read == 23:
            [num_samples] = np.frombuffer(pkt_hdr[19:23], dtype=np.uint32)
            pkt_length = header_length + 2 * num_samples

        # load into buffer, resizing as necessary
        if len(self.buffer) < pkt_length:
            self.buffer.resize(pkt_length, refcheck=False)
        n_bytes_read = self.in_stream.readinto(self.buffer[header_length:pkt_length])
        self.n_bytes_read += n_bytes_read
        if n_bytes_read != pkt_length - header_length:
            raise RuntimeError(
                f"only got {n_bytes_read} bytes for packet read when {pkt_length-header_length} were expected."
            )

        # return just the packet
        return self.buffer[:pkt_length]

    def read_packet(self) -> bool:
        """Read and decode a packet of data.
        Data written to the `event_rbkd` attribute.
        """
        packet = self.load_packet()
        # Return True until we have no more file left to read
        if packet is None:
            return False
        self.packet_id += 1
        self.any_full |= self.event_decoder.decode_packet(
            packet, self.event_rbkd, self.packet_id, self.header
        )

        return True
