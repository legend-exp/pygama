'''
orca packets are ndarrays of type uint32

1 'word' is one uint32
'''

def is_short(packet):
    return bool(packet[0] >> 31)

def get_n_words(packet):
    if is_short(packet): return 1
    return packet[0] & 0x3FFFF

def get_data_id(packet, shift=True):
    if is_short(packet):
        if shift: return (packet[0] & 0xFC000000) >> 26
        else: return packet[0] & 0xFC000000
    if shift: return (packet[0] & 0xFFFC0000) >> 18
    return packet[0] & 0xFFFC0000

