import logging
from typing import Any

from pygama.raw.orca.orca_base import OrcaDecoder
from pygama.raw.orca.orca_header import OrcaHeader
from pygama.raw.orca.orca_packet import OrcaPacket
from pygama.raw.raw_buffer import RawBufferLibrary

log = logging.getLogger(__name__)


class ORRunDecoderForRun(OrcaDecoder):
    """Decoder for Run Control data written by ORCA."""

    def __init__(self, header: OrcaHeader = None, **kwargs) -> None:
        # quickstartrun through startsubrunrecord are in order
        # of increasing bit location with respect to the rightmost
        # bit in data packet
        self.decoded_values = {
            "subrun_number": {"dtype": "uint16"},
            "runstartorstop": {"dtype": "bool8"},
            "quickstartrun": {"dtype": "bool8"},
            "remotecontrolrun": {"dtype": "bool8"},
            "heartbeatrecord": {"dtype": "bool8"},
            "endsubrunrecord": {"dtype": "bool8"},
            "startsubrunrecord": {"dtype": "bool8"},
            "run_number": {"dtype": "int32"},
            "time": {"dtype": "int32"},
        }
        super().__init__(header=header, **kwargs)

    def get_decoded_values(self, key: int = None) -> dict[str, Any]:
        return self.decoded_values

    def decode_packet(
        self, packet: OrcaPacket, packet_id: int, rbl: RawBufferLibrary
    ) -> bool:
        """
        Decode the ORCA Run packet.

        Format is taken from the ORCA website: `Run Control
        <http://orca.physics.unc.edu/orca/Data_Chain/Run_Control.html>`_

        """

        if len(rbl) != 1:
            log.warning(
                f"got {len(rbl)} rb's, should have only 1 (no keyed decoded values)"
            )
        rb = rbl[0]
        tbl = rb.lgdo
        ii = rb.loc

        tbl["subrun_number"].nda[ii] = (packet[1] & 0xFFFF0000) >> 16
        for i, k in enumerate(self.decoded_values):
            if 0 < i < 7:
                tbl[k].nda[ii] = (packet[1] & (1 << (i - 1))) >> (i - 1)

        tbl["run_number"].nda[ii] = packet[2]
        tbl["time"].nda[ii] = packet[3]

        rb.loc += 1
        return rb.is_full()
