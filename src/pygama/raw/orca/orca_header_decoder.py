from ..data_decoder import *
from pygama import lgdo
import plistlib

class OrcaHeaderDecoder(DataDecoder):
    """
    Decodes Orca headers
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.header = lgdo.Struct()

    def make_lgdo(self, key=None, size=None): return self.header

    def buffer_is_full(self, rb): return rb.loc > 0

    def decode_packet(self, packet, packet_id, rbl=None, verbosity=0):
        if len(self.header.keys()) != 0:
            print("Warning: got a second header...")
        if packet_id != 0:
            print(f"Warning: header was packet {packet_id}")
        n_bytes = packet[1] # word 1 is the header packet length in bytes
        as_bytes = packet[2:].tobytes()
        header_dict = plistlib.loads(as_bytes[:n_bytes], fmt=plistlib.FMT_XML)
        self.header.update(header_dict)
        return True # header is "full" 

    def get_decoder_list(self):
        decoder_names = []
        dd = self.header['dataDescription']
        for class_key in dd.keys():
            for super_key in dd[class_key].keys():
                decoder_names.append(dd[class_key][super_key]['decoder'])
        return decoder_names

    def get_data_id(self, decoder_name):
        dd = header_dict['dataDescription']
        for class_key in dd.keys():
            for super_key in dd[class_key].keys():
                if dd[class_key][super_key]['decoder'] == decoder_name:
                    return dd[class_key][super_key]['dataId']
