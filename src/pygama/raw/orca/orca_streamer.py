from __future__ import annotations

import gzip
import json
import logging

import numpy as np

from pygama.raw.data_streamer import DataStreamer
from pygama.raw.orca import orca_packet
from pygama.raw.orca.orca_base import OrcaDecoder
from pygama.raw.orca.orca_digitizers import (  # noqa: F401
    ORSIS3302DecoderForEnergy,
    ORSIS3316WaveformDecoder,
)
from pygama.raw.orca.orca_flashcam import (  # noqa: F401
    ORFlashCamADCWaveformDecoder,
    ORFlashCamListenerConfigDecoder,
    ORFlashCamWaveformDecoder,
)
from pygama.raw.orca.orca_header_decoder import OrcaHeaderDecoder
from pygama.raw.orca.orca_run_decoder import ORRunDecoderForRun  # noqa: F401
from pygama.raw.raw_buffer import RawBuffer, RawBufferLibrary

log = logging.getLogger(__name__)


class OrcaStreamer(DataStreamer):
    """Data streamer for ORCA data."""

    def __init__(self) -> None:
        super().__init__()
        self.in_stream = None
        self.buffer = np.empty(1024, dtype="uint32")  # start with a 4 kB packet buffer
        self.header = None
        self.header_decoder = OrcaHeaderDecoder()
        self.decoder_id_dict = {}  # dict of data_id to decoder object
        self.rbl_id_dict = {}  # dict of RawBufferLists for each data_id

    # TODO: need to correct for endianness?
    def load_packet(self, skip_unknown_ids: bool = False) -> np.uint32 | None:
        """Loads the next packet into the internal buffer.

        Returns packet as a :class:`numpy.uint32` view of the buffer (a slice),
        returns ``None`` at EOF.
        """
        if self.in_stream is None:
            raise RuntimeError("self.in_stream is None")

        # read packet header
        pkt_hdr = self.buffer[:1]
        n_bytes_read = self.in_stream.readinto(pkt_hdr)  # buffer is at least 4 kB long
        self.n_bytes_read += n_bytes_read
        if n_bytes_read == 0:
            return None
        if n_bytes_read != 4:
            raise RuntimeError(f"only got {n_bytes_read} bytes for packet header")

        # if it's a short packet, we are done
        if orca_packet.is_short(pkt_hdr):
            return pkt_hdr

        # long packet: get length and check if we can skip it
        n_words = orca_packet.get_n_words(pkt_hdr)
        if (
            skip_unknown_ids
            and orca_packet.get_data_id(pkt_hdr, shift=False)
            not in self.decoder_id_dict
        ):
            self.in_stream.seek((n_words - 1) * 4, 1)
            self.n_bytes_read += (n_words - 1) * 4  # well, we didn't really read it...
            return pkt_hdr

        # load into buffer, resizing as necessary
        if len(self.buffer) < n_words:
            self.buffer.resize(n_words, refcheck=False)
        n_bytes_read = self.in_stream.readinto(self.buffer[1:n_words])
        self.n_bytes_read += n_bytes_read
        if n_bytes_read != (n_words - 1) * 4:
            raise RuntimeError(
                f"only got {n_bytes_read} bytes for packet read when {(n_words-1)*4} were expected."
            )

        # return just the packet
        return self.buffer[:n_words]

    def get_decoder_list(self) -> list[OrcaDecoder]:
        return list(self.decoder_id_dict.values())

    def set_in_stream(self, stream_name: str) -> None:
        if self.in_stream is not None:
            self.close_in_stream()
        if stream_name.endswith(".gz"):
            self.in_stream = gzip.open(stream_name.encode("utf-8"), "rb")
        else:
            self.in_stream = open(stream_name.encode("utf-8"), "rb")
        self.n_bytes_read = 0

    def close_in_stream(self) -> None:
        if self.in_stream is None:
            raise RuntimeError("tried to close an unopened stream")
        self.in_stream.close()
        self.in_stream = None

    def close_stream(self) -> None:
        self.close_in_stream()

    def is_orca_stream(stream_name: str) -> bool:  # noqa: N805
        orca = OrcaStreamer()
        orca.set_in_stream(stream_name)
        first_bytes = orca.in_stream.read(12)
        orca.close_in_stream()

        # that read should have succeeded
        if len(first_bytes) != 12:
            return False

        # first 14 bits should be zero
        uints = np.frombuffer(first_bytes, dtype="uint32")
        if (uints[0] & 0xFFFC0000) != 0:
            return False

        # xml header length should fit within header packet length
        pad = uints[0] * 4 - 8 - uints[1]
        if pad < 0 or pad > 3:
            return False

        # last 4 chars should be '<?xm'
        if first_bytes[8:].decode() != "<?xm":
            return False

        # it must be an orca stream
        return True

    def hex_dump(
        self,
        stream_name: str,
        n_packets: int = np.inf,
        skip_header: bool = False,
        shift_data_id: bool = True,
        print_n_words: bool = False,
        max_words: int = np.inf,
        as_int: bool = False,
        as_short: bool = False,
    ) -> None:
        self.set_in_stream(stream_name)
        if skip_header:
            self.load_packet()
        while n_packets > 0:
            packet = self.load_packet()
            if packet is None:
                self.close_in_stream()
                return
            orca_packet.hex_dump(
                packet,
                shift_data_id=shift_data_id,
                print_n_words=print_n_words,
                max_words=max_words,
                as_int=as_int,
                as_short=as_short,
            )
            n_packets -= 1

    def open_stream(
        self,
        stream_name: str,
        rb_lib: RawBufferLibrary = None,
        buffer_size: int = 8192,
        chunk_mode: str = "any_full",
        out_stream: str = "",
    ) -> list[RawBuffer]:
        """Initialize the ORCA data stream.

        Parameters
        ----------
        stream_name
            The ORCA filename. Only file streams are currently supported.
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
            a list of length 1 containing the raw buffer holding the ORCA header.
        """

        self.set_in_stream(stream_name)

        # read in the header
        packet = self.load_packet()
        if orca_packet.get_data_id(packet) != 0:
            raise RuntimeError(
                f"got data id {orca_packet.get_data_id(packet)} for header"
            )

        self.packet_id = 0
        self.any_full |= self.header_decoder.decode_packet(packet, self.packet_id)
        self.header = self.header_decoder.header

        # find the names of all decoders listed in the header AND in the rb_lib (if specified)
        decoder_names = self.header.get_decoder_list()
        if rb_lib is not None and "*" not in rb_lib:
            keep_decoders = []
            for name in decoder_names:
                if name in rb_lib:
                    keep_decoders.append(name)
            decoder_names = keep_decoders
            # check that all requested decoders are present
            for name in rb_lib.keys():
                if name not in keep_decoders:
                    log.warning(
                        f"decoder {name} (requested in rb_lib) not in data description in header"
                    )

        # get a mapping of data_ids-of-interest to instantiated decoders
        id_to_dec_name_dict = self.header.get_id_to_decoder_name_dict(
            shift_data_id=False
        )
        instantiated_decoders = {"OrcaHeaderDecoder": self.header_decoder}
        for data_id in id_to_dec_name_dict.keys():
            name = id_to_dec_name_dict[data_id]
            if name not in instantiated_decoders:
                if name not in globals():
                    log.warning(
                        f"no implementation of {name}, corresponding packets will be skipped"
                    )
                    continue
                decoder = globals()[name]
                instantiated_decoders[name] = decoder(header=self.header)
            self.decoder_id_dict[data_id] = instantiated_decoders[name]

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
        good_buffers = []
        for data_id in self.decoder_id_dict.keys():
            name = id_to_dec_name_dict[data_id]
            if name not in self.rb_lib:
                log.info(f"skipping data from {name}")
                continue
            self.rbl_id_dict[data_id] = self.rb_lib[name]
            good_buffers.append(name)
        # check that we have instantiated decoders for all buffers
        for key in self.rb_lib:
            if key not in good_buffers:
                log.warning(f"buffer for {key} has no decoder")
        log.debug(f"rb_lib = {self.rb_lib}")

        # return header raw buffer
        if "OrcaHeaderDecoder" in rb_lib:
            header_rb_list = rb_lib["OrcaHeaderDecoder"]
            if len(header_rb_list) != 1:
                log.warning(
                    f"header_rb_list had length {len(header_rb_list)}, ignoring all but the first"
                )
            rb = header_rb_list[0]
        else:
            rb = RawBuffer(lgdo=self.header_decoder.make_lgdo())
        rb.lgdo.value = json.dumps(self.header)
        rb.loc = 1  # we have filled this buffer
        return [rb]

    def read_packet(self) -> bool:
        """Read a packet of data.

        Data written to the `rb_lib` attribute.
        """
        # read until we get a decodeable packet
        while True:
            packet = self.load_packet(skip_unknown_ids=True)
            if packet is None:
                return False
            self.packet_id += 1

            # look up the data id, decoder, and rbl
            data_id = orca_packet.get_data_id(packet, shift=False)
            log.debug(
                f"packet {self.packet_id}: data_id = {data_id}, decoder = {'None' if data_id not in self.decoder_id_dict else type(self.decoder_id_dict[data_id]).__name__}"
            )
            if data_id in self.rbl_id_dict:
                break

        # now decode
        decoder = self.decoder_id_dict[data_id]
        rbl = self.rbl_id_dict[data_id]
        self.any_full |= decoder.decode_packet(packet, self.packet_id, rbl)
        return True
