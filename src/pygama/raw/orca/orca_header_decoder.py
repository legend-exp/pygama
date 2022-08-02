from __future__ import annotations

import json
import logging
import plistlib

from pygama import lgdo
from pygama.raw.orca.orca_base import OrcaDecoder
from pygama.raw.orca.orca_header import OrcaHeader
from pygama.raw.orca.orca_packet import OrcaPacket
from pygama.raw.raw_buffer import RawBuffer, RawBufferLibrary

log = logging.getLogger(__name__)


class OrcaHeaderDecoder(OrcaDecoder):
    """Decodes ORCA headers."""

    def __init__(self, header: OrcaHeader = None, **kwargs) -> None:
        if header is not None:
            log.warning("unexpected non-None header")
        super().__init__(header=header, **kwargs)
        self.header = OrcaHeader()

    def make_lgdo(self, key: int = None, size: int = None) -> lgdo.Scalar:
        return lgdo.Scalar(value="")

    def buffer_is_full(self, rb: RawBuffer) -> bool:
        return rb.loc > 0

    def decode_packet(
        self, packet: OrcaPacket, packet_id: int, rbl: RawBufferLibrary = None
    ) -> bool:
        if len(self.header.keys()) != 0:
            log.warning("got an unexpected second header")
        if packet_id != 0:
            log.warning(f"header was packet number {packet_id}")
        n_bytes = packet[1]  # word 1 is the header packet length in bytes
        as_bytes = packet[2:].tobytes()
        self.header.update(plistlib.loads(as_bytes[:n_bytes], fmt=plistlib.FMT_XML))
        if rbl is not None:
            if len(rbl) != 1 or not isinstance(rbl[0].lgdo, lgdo.Scalar):
                raise ValueError(
                    f"invalid RawBufferLibrary object: len(rbl) = {len(rbl)}, type = {type(rbl[0].lgdo)}"
                )
            else:
                rbl[0].lgdo.value = json.dumps(self.header)
                rbl[0].loc = 1
        return True  # header raw buffer is "full"
