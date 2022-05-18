import plistlib

from pygama import lgdo

from ..data_decoder import *
from . import orca_header


class OrcaHeaderDecoder(DataDecoder):
    """
    Decodes Orca headers
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.header = orca_header.OrcaHeader()

    def make_lgdo(self, key=None, size=None): return lgdo.Scalar(value='')

    def buffer_is_full(self, rb): return rb.loc > 0

    def decode_packet(self, packet, packet_id, rbl=None, verbosity=0):
        if len(self.header.keys()) != 0:
            print("Warning: got a second header...")
        if packet_id != 0:
            print(f"Warning: header was packet {packet_id}")
        n_bytes = packet[1] # word 1 is the header packet length in bytes
        as_bytes = packet[2:].tobytes()
        self.header.update(plistlib.loads(as_bytes[:n_bytes], fmt=plistlib.FMT_XML))
        if rbl is not None:
            if len(rbl) != 1 or not isinstance(rbl[0].lgdo, lgdo.Scalar):
                print(f"Error: len(rbl) = {len(rbl)}, type = {type(rbl[0].lgdo)}")
            else:
                rbl[0].lgdo.value = json.dumps(self.header)
                rbl[0].loc = 1
        return True # header raw buffer is "full"
