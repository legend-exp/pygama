import gzip
import numpy as np

from ..data_streamer import DataStreamer
from . import orca_packet
from .orca_header_decoder import OrcaHeaderDecoder

class OrcaStreamer(DataStreamer):
    """ Data streamer for ORCA data
    """
    def __init__(self):
        super().__init__()
        self.in_stream = None
        self.buffer = np.empty(1024, dtype='uint32') # start with a 4 kB buffer
        self.header_decoder = OrcaHeaderDecoder()
        self.decoder_id_dict = {} # dict of data_id to decoder object
        self.decoder_name_dict = {} # dict of name to decoder object
        self.rbl_id_dict = {} # dict of RawBufferLists for each data_id


    def load_packet(self, skip_unknown_ids=False):
        """ Loads the next packet into the internal buffer
        Returns packet as a uint32 view of the buffer (a slice)
        Returns None at EOF or for an error

        CHECK: need to correct for endianness?
        """
        if self.in_stream is None:
            print('Error: in_stream is None')
            return None

        # read packet header
        pkt_hdr = self.buffer[:1]
        n_bytes_read = self.in_stream.readinto(pkt_hdr) # buffer is at least 4 kB long
        self.n_bytes_read += n_bytes_read
        if n_bytes_read == 0: return None
        if n_bytes_read != 4:
            print(f'Error: only got {n_bytes_read} bytes for packet header')
            return None

        # if it's a short packet, we are done
        if orca_packet.is_short(pkt_hdr): return pkt_hdr

        # long packet: get length and check if we can skip it
        n_words = orca_packet.get_n_words(pkt_hdr)
        if skip_unknown_ids and orca_packet.get_data_id(pkt_hdr) not in self.decoder_id_dict:
            self.in_stream.seek((n_words-1)*4, 1)
            self.n_bytes_read += (n_words-1)*4 # well, we didn't really read it... 
            return pkt_hdr

        # load into buffer, resizing as necessary
        if len(self.buffer) < n_words: self.buffer.resize(n_words, refcheck=False)
        n_bytes_read = self.in_stream.readinto(self.buffer[1:n_words])
        self.n_bytes_read += n_bytes_read
        if n_bytes_read != (n_words-1)*4:
            print(f'Error: only got {n_bytes_read} bytes for packet read when {(n_words-1)*4} were expected.')
            return None

        # return just the packet
        return self.buffer[:n_words]


    def get_decoder_list(self):
        return list(self.decoder_id_dict.values())


    def set_in_stream(self, stream_name):
        if self.in_stream is not None: self.close_in_stream()
        if stream_name.endswith('.gz'):
            self.in_stream = gzip.open(stream_name.encode('utf-8'), 'rb')
        else: self.in_stream = open(stream_name.encode('utf-8'), 'rb')
        self.n_bytes_read = 0


    def close_in_stream(self):
        self.in_stream.close()
        self.in_stream = None


    def is_orca_stream(stream_name): # static function
        orca = OrcaStreamer()
        orca.set_in_stream(stream_name)
        first_bytes = orca.in_stream.read(12)

        # that read should have succeeded
        if len(first_bytes) != 12: return False

        # first 14 bits should be zero
        uints = np.frombuffer(first_bytes, dtype='uint32')
        if (uints[0] & 0xfffc0000) != 0: return False

        # xml header length should fit within header packet length
        pad = uints[0] * 4 - 8 - uints[1]
        if pad < 0 or pad > 3: return False

        # last 4 chars should be '<?xm'
        if first_bytes[8:].decode() != '<?xm': return False

        # it must be an orca stream
        return True


    def hex_dump(self, stream_name, n_packets=np.inf, 
                 skip_header=False, shift_data_id=True, print_n_words=False, 
                 max_words=np.inf, as_int=False, as_short=False):
        self.set_in_stream(stream_name)
        if skip_header: self.load_packet()
        while n_packets > 0:
            packet = self.load_packet()
            if packet is None: 
                self.close_in_stream()
                return
            data_id = orca_packet.get_data_id(packet, shift=shift_data_id)
            n_words = orca_packet.get_n_words(packet)
            if print_n_words: print(f'data ID = {data_id}: {n_words} words')
            else:
                print(f'data ID = {data_id}:')
                n_to_print = int(np.minimum(n_words, max_words))
                pad = int(np.ceil(np.log10(n_to_print)))
                for i in range(n_to_print):
                    line = f'{str(i).zfill(pad)}'
                    line += ' {0:#0{1}x}'.format(packet[i], 10)
                    if data_id == 0 and i > 1: line += f' {packet[i:i+1].tobytes().decode()}'
                    if as_int: line += f' {packet[i]}'
                    if as_short: line += f" {np.frombuffer(packet[i:i+1].tobytes(), dtype='uint16')}"
                    print(line)
            n_packets -= 1


    def open_stream(self, stream_name, rb_lib=None, buffer_size=8192,
                    chunk_mode='any_full', out_stream='', verbosity=0):
        """ Initialize the ORCA data stream

        Parameters
        ----------
        stream_name : str
            The ORCA filename.  Only file streams are currently supported.
            Socket stream reading can be added later.
        rb_lib : RawBufferLibrary
            library of buffers for this stream
        buffer_size : int
            length of tables to be read out in read_chunk
        verbosity : int
            verbosity level for the initialize function
 
        Returns
        -------
        header_data : list(RawBuffer)
            a list of length 1 containing the raw buffer holding the ORCA header
        """
        
        self.set_in_stream(stream_name)

        # read in the header
        packet = self.load_packet()
        if orca_packet.get_data_id(packet) != 0:
            print(f'Error: got data id {orca_packet.get_data_id(packet)} for header')
            return []
        self.packet_id = 0
        self.any_full |= self.header_decoder.decode_packet(packet, self.packet_id, verbosity=verbosity)

        # instantiate decoders listed in the header AND in the rb_lib (if specified)
        decoder_names = ['OrcaHeaderDecoder']
        decoder_names += self.header_decoder.get_decoder_list()
        if rb_lib is not None and '*' not in rb_lib:
            keep_decoders = []
            for name in decoder_names:
                if name in rb_lib: keep_decoders.append[name]
            decoder_names = keep_decoders
            # check that all requested decoders are present
            for name in rb_lib.keys():
                if name not in keep_decoders:
                    print(f'Warning: decoder {name} (requested in rb_lib) not in data description in header')
        for name in decoder_names:
            # handle header decoder specially
            if name == 'OrcaHeaderDecoder':
                self.decoder_id_dict[0] = self.header_decoder
                self.decoder_name_dict['OrcaHeaderDecoder'] = 0
                continue
            # instantiate other decoders by name
            if name not in globals():
                print(f'Warning: No implementation of {name}, corresponding packets will be skipped')
                continue
            decoder = globals()[name]
            decoder.data_id = self.header_decoder.get_data_id(name)
            self.decoder_id_dict[decoder.data_id] = decoder
            self.decoder_name_dict[name] = decoder.data_id

        # initialize the buffers in rb_lib. Store them for fast lookup
        super().open_stream(stream_name, rb_lib, buffer_size=buffer_size,
                            chunk_mode=chunk_mode, out_stream=out_stream, verbosity=verbosity)
        if rb_lib is None: rb_lib = self.rb_lib
        for name in self.rb_lib.keys(): 
            data_id = self.decoder_name_dict[name]
            self.rbl_id_dict[data_id] = self.rb_lib[name]
        print(self.rb_lib)

        # return header raw buffer
        if 'OrcaHeaderDecoder' in rb_lib: 
            header_rb_list = rb_lib['OrcaHeaderDecoder']
            if len(header_rb_list) != 1:
                print(f'warning! header_rb_list had length {len(header_rb_list)}, ignoring all but the first')
            rb = header_rb_list[0]
        else: rb = RawBuffer(lgdo=self.header_decoder.header)
        rb.loc = 1 # we have filled this buffer
        return [rb]


    def read_packet(self, verbosity=0):
        """ Read a packet of data.

        Data written to self.rb_lib.
        """
        # read until we get a decodeable packet
        while True:
            packet = self.load_packet(skip_unknown_ids=True)
            if packet is None: return False
            self.packet_id += 1

            # look up the data id, decoder, and rbl
            data_id = orca_packet.get_data_id(packet)
            if verbosity>0: print(f'packet {self.packet_id}: data_id = {data_id}')
            if data_id in self.decoder_id_dict: break

        # now decode
        decoder = self.decoder_id_dict[data_id]
        rbl = self.rbl_id_dict[data_id]
        self.any_full |= decoder.decode_packet(packet, self.packet_id, rbl, verbosity=verbosity)
        return True



'''
import sys, gzip
import numpy as np
import plistlib

from tqdm.std import tqdm
from ..utils import tqdm_range, update_progress
from .io_base import DataDecoder
from pygama import lh5
from .ch_group import *


class OrcaDecoder(DataDecoder):
    """ Base class for ORCA decoders.

    ORCA data packets have a dataID-to-decoder_name mapping so these decoders
    need to have self.decoder_name defined in __init__

    ORCA also stores an object_info dictionary in the header by 'class name" so
    these decoders need to have self.orca_class_name defined in __init__

    ORCA also uses a uniform packet structure so put some boiler plate here so
    that all ORCA decoders can make use of it.
    """
    def __init__(self, dataID=None, object_info=[], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataID = dataID
        self.set_object_info(object_info)

    def set_header_dict(self, header_dict):
        self.header_dict = header_dict
        self.set_object_info(get_object_info(header_dict, self.orca_class_name))

    def set_object_info(self, object_info):
        """Overload to e.g. update decoded_values based on object_info."""
        self.object_info = object_info


def open_orca(orca_filename):
    if orca_filename.endswith('.gz'):
        return gzip.open(orca_filename.encode('utf-8'), 'rb')
    else: return open(orca_filename.encode('utf-8'), 'rb')


def parse_header(orca_filename):
    """
    Opens the given file for binary read ('rb'), then grabs the first 8 bytes
    The first 4 bytes (1 long) of an orca data file are the total length in
    longs of the record
    The next 4 bytes (1 long) is the length of the header in bytes
    The header is then read in ...
    """
    with open_orca(orca_filename) as xmlfile_handle:
        #read the first word:
        ba = bytearray(xmlfile_handle.read(8))

        #Replacing this to be python2 friendly
        # #first 4 bytes: header length in long words
        # i = int.from_bytes(ba[:4], byteorder=sys.byteorder)
        # #second 4 bytes: header length in bytes
        # j = int.from_bytes(ba[4:], byteorder=sys.byteorder)

        big_endian = False if sys.byteorder == "little" else True
        i = from_bytes(ba[:4], big_endian=big_endian)
        j = from_bytes(ba[4:], big_endian=big_endian)
        if (np.ceil(j/4) != i-2) and (np.ceil(j/4) != i-3):
            print('Error: header byte length = %d is the wrong size to fit into %d header packet words' % (j, i-2))
            return i, j, {}

        #read in the next that-many bytes that occupy the plist header
        as_bytes = xmlfile_handle.read(j)
        ba = bytearray(as_bytes)

        #convert to string
        #the readPlistFromBytes method doesn't exist in 2.7
        if sys.version_info[0] < 3:
            header_string = ba.decode("utf-8")
            header_dict = plistlib.readPlistFromString(header_string)
        elif sys.version_info[1] < 9:
            header_dict = plistlib.readPlistFromBytes(ba)
        else:
            header_dict = plistlib.loads(as_bytes, fmt=plistlib.FMT_XML)
        return i, j, header_dict


def from_bytes(data, big_endian=False):
    """
    python2 doesn't have this function,
    so rewrite it for backwards compatibility
    """
    if isinstance(data, str):
        data = bytearray(data)
    if big_endian:
        data = reversed(data)
    num = 0
    for offset, byte in enumerate(data):
        num += byte << (offset * 8)
    return num


def get_run_number(header_dict):
    """ header_dict parse functions, ORCA specific """
    for d in (header_dict["ObjectInfo"]["DataChain"]):
        if "Run Control" in d:
            return (d["Run Control"]["RunNumber"])
    raise ValueError("No run number found in header!")


def get_data_id(header_dict, class_name, super_name):
    """
    stored like this:
    `header_dict["dataDescription"]["ORRunModel"]["Run"]["dataId"]`
    but integer needs to be bitshifted by 18
    """
    id_int = header_dict["dataDescription"][class_name][super_name]["dataId"]

    return id_int >> 18


def flip_data_ids(header_dict):
    """
    Returns an inverted dictionary such that:
    Could be extended somehow to give you all the supers associated with a given class name (maybe like)
    flipped[dataId] = [class_key, [super1, super2, ...]]
    """
    flipped = dict()
    # header_dict["dataDescription"][class_name][super_name]["dataId"]
    for class_key in header_dict["dataDescription"].keys():
        super_keys_list = []
        for super_key in header_dict["dataDescription"][class_key].keys():
            super_keys_list.append(super_key)
            ID_val = (header_dict["dataDescription"][class_key][super_key]
                      ["dataId"]) >> 18
            flipped[ID_val] = [class_key, super_keys_list]

    # this one just gives a single super             flipped[dataId] = [class_key, super_key]
    # for class_key in header_dict["dataDescription"].keys():
    #     super_keys_list = header_dict["dataDescription"][class_key].keys()
    #     ID_val = (header_dict["dataDescription"][class_key][super_keys_list[0]]["dataId"])>>18
    #     flipped[ID_val] = [class_key,super_keys_list]

    return flipped


def get_id_to_decoder_name_dict(header_dict):
    """
    Returns a dictionary that goes:
    `dict[dataID] = "decoderName"`
    e.g: d[5] = 'ORSIS3302DecoderForEnergy'
    """
    id2dn_dict = {}
    dd = header_dict['dataDescription']
    for class_key in dd.keys():
        for super_key in dd[class_key].keys():
            dataID = (dd[class_key][super_key]['dataId']) >> 18
            decoder_name = dd[class_key][super_key]['decoder']
            id2dn_dict[dataID] = decoder_name
    return id2dn_dict


def get_object_info(header_dict, orca_class_name):
    """
    returns a list with all info from the header for each card with name
    orca_class_name.
    """
    object_info_list = []

    crates = header_dict["ObjectInfo"]["Crates"]
    for crate in crates:
        cards = crate["Cards"]
        for card in cards:
            if card["Class Name"] == orca_class_name:
                card["Crate"] = crate["CrateNumber"]
                object_info_list.append(card)

    if len(object_info_list) == 0:
        print('OrcaDecoder::get_object_info(): Warning: no object info '
              'for class name', orca_class_name)
    return object_info_list


def get_readout_info(header_dict, orca_class_name, unique_id=-1):
    """
    retunrs a list with all the readout list info from the header with name
    orca_class_name.  optionally, if unique_id >= 0 only return the list for
    that Orca unique id number.
    """
    object_info_list = []
    try:
        readouts = header_dict["ReadoutDescription"]
        for readout in readouts:
            try:
                if readout["name"] == orca_class_name:
                    if unique_id >= 0:
                        if obj["uniqueID"] != unique_id: continue
                    object_info_list.append(readout)
            except KeyError: continue
    except KeyError: pass
    if len(object_info_list) == 0:
        print('OrcaDecoder::get_readout_info(): warning: no readout info '
              'for class name', orca_class_name)
    return object_info_list


def get_auxhw_info(header_dict, orca_class_name, unique_id=-1):
    """
    returns a list with all the info from the AuxHw table of the header
    with name orca_class_name.  optionally, if unique_id >= 0 only return
    the object for that Orca unique id number.
    """
    object_info_list = []
    try:
        objs = header_dict["ObjectInfo"]["AuxHw"]
        for obj in objs:
            try:
                if obj["Class Name"] == orca_class_name:
                    if unique_id >= 0:
                        if obj["uniqueID"] != unique_id: continue
                    object_info_list.append(obj)
            except KeyError: continue
    except KeyError: pass
    if len(object_info_list) == 0:
        print('OrcaDecoder::get_auxhw_info(): warning: no object info '
              'for class name', orca_class_name)
    return object_info_list


def get_next_packet(f_in):
    """
    Gets the next packet, and some basic information about it
    Takes the file pointer as input
    Outputs:
    - event_data: a byte array of the data produced by the card (could be header + data)
    - data_id: This is the identifier for the type of data-taker (i.e. Gretina4M, etc)
    - crate: the crate number for the packet
    - card: the card number for the packet
    # number of bytes to read in = 8 (2x 32-bit words, 4 bytes each)
    # The read is set up to do two 32-bit integers, rather than bytes or shorts
    # This matches the bitwise arithmetic used elsewhere best, and is easy to implement
    """
    try:
        # event header is 8 bytes (2 longs)
        head = np.fromstring(f_in.read(4), dtype=np.uint32)
    except Exception as e:
        print(e)
        raise Exception("Failed to read in the event orca header.")

    # Assuming we're getting an array of bytes:
    # record_length   = (head[0] + (head[1]<<8) + ((head[2]&0x3)<<16))
    # data_id         = (head[2] >> 2) + (head[3]<<8)
    # card            = (head[6] & 0x1f)
    # crate           = (head[6]>>5) + head[7]&0x1
    # reserved        = (head[4] + (head[5]<<8))

    # Using an array of uint32
    record_length = int((head[0] & 0x3FFFF))
    data_id = int((head[0] >> 18))
    # reserved =int( (head[1] &0xFFFF))

    # /* ========== read in the rest of the event data ========== */
    try:
        # record_length is in longs, read gives bytes
        event_data = f_in.read(record_length * 4 - 4)
    except Exception as e:
        print("  No more data...\n")
        print(e)
        raise EOFError

    return event_data, data_id


def get_ccc(crate, card, channel):
    return (crate << 9) + ((card & 0x1f) << 4) + (channel & 0xf)


def get_crate(ccc):
    return ccc >> 9


def get_card(ccc):
    return (ccc >> 4) & 0x1f


def get_channel(ccc):
    return ccc & 0xf


# Import orca_digitizers so that the list of OrcaDecoder.__subclasses__ gets populated
# Do it here so that orca_digitizers can import the functions above here
from . import orca_digitizers, orca_flashcam

def process_orca(daq_filename, raw_file_pattern, n_max=np.inf, ch_groups_dict=None, verbose=False, buffer_size=1024):
    """
    convert ORCA DAQ data to "raw" lh5

    ch_groups_dict: keyed by decoder_name
    """
    lh5_store = lh5.Store()

    f_in = open_orca(daq_filename)
    if f_in == None:
        print("Couldn't find the file %s" % daq_filename)
        sys.exit(0)

    # parse the header. save the length so we can jump past it later
    reclen, header_nbytes, header_dict = parse_header(daq_filename)

    # figure out the total size
    SEEK_END = 2
    f_in.seek(0, SEEK_END)
    file_size = float(f_in.tell())
    f_in.seek(0, 0)  # rewind
    file_size_MB = file_size / 1e6
    print("Total file size: {:.3f} MB".format(file_size_MB))
    print("Run number:", get_run_number(header_dict))


    # Build the dict used in the inner loop for passing data packets to decoders
    decoders = {}

    # First build a list of all decoder names that might be in the data
    # This is a dict of names keyed off of data_id
    id2dn_dict = get_id_to_decoder_name_dict(header_dict)
    if verbose:
        print("Data IDs present in ORCA file header are:")
        for data_id in id2dn_dict:
            print(f"    {data_id}: {id2dn_dict[data_id]}")

    # Invert the previous list, to get a list of decoder ids keyed off of
    # decoder names
    dn2id_dict = {name:data_id for data_id, name in id2dn_dict.items()}

    # By default we decode all data for which we have decoders. If the user
    # provides a ch_group_dict, we will only decode data from decoders keyed in
    # the dict.
    decode_all_data = True
    decoders_to_run = dn2id_dict.keys()
    if ch_groups_dict is not None:
        decode_all_data = False
        decoders_to_run = ch_groups_dict.keys()

    # Now get the actual requested decoders
    for sub in OrcaDecoder.__subclasses__():
        decoder = sub() # instantiate the class
        if decoder.decoder_name in decoders_to_run:
            decoder.dataID = dn2id_dict[decoder.decoder_name]
            decoder.set_header_dict(header_dict)
            decoders[decoder.dataID] = decoder
    if len(decoders) == 0:
        print("No decoders. Exiting...")
        sys.exit(1)
    if verbose:
        print("pygama will run these decoders:")
        for data_id, dec in decoders.items():
            print("   ", dec.decoder_name+ ", id =", data_id)

    # Now cull the decoders_to_run list
    new_dtr = []
    for decoder_name in decoders_to_run:
        data_id = dn2id_dict[decoder_name]
        if data_id not in decoders.keys():
            print("warning: no decoder exists for", decoder_name, "... will skip its data.")
        else: new_dtr.append(decoder_name)
    decoders_to_run = new_dtr

    # prepare ch groups
    if ch_groups_dict is None:
        ch_groups_dict = {}
        for decoder_name in decoders_to_run:
            ch_groups = create_dummy_ch_groups()
            ch_groups_dict[decoder_name] = ch_groups
            grp_path_template = f'{decoder_name}/raw'
            set_outputs(ch_groups, out_file_template=raw_file_pattern, grp_path_template=grp_path_template)
    else:
        for decoder_name, ch_groups in ch_groups_dict.items():
            expand_ch_groups(ch_groups)
            set_outputs(ch_groups, out_file_template=raw_file_pattern, grp_path_template='{system}/{group_name}/raw')

    # Set up tables for data
    ch_tables_dict = {}
    for data_id, dec in decoders.items():
        decoder_name = id2dn_dict[data_id]
        ch_groups = ch_groups_dict[decoder_name]
        ch_tables_dict[data_id] = build_tables(ch_groups, buffer_size, dec)
    max_tbl_size = 0

    # -- scan over raw data --
    print("Beginning daq-to-raw processing ...")

    packet_id = 0  # number of events decoded
    unrecognized_data_ids = []

    # skip the header using reclen from before
    # reclen is in number of longs, and we want to skip a number of bytes
    f_in.seek(reclen * 4)

    n_entries = 0
    unit = "B"
    if n_max < np.inf and n_max > 0:
        n_entries = n_max
        unit = "id"
    else:
        n_entries = file_size
    progress_bar = tqdm_range(0, int(n_entries), text="Processing", verbose=verbose, unit=unit)
    file_position = 0

    # start scanning
    while (packet_id < n_max and f_in.tell() < file_size):
        packet_id += 1

        try:
            packet, data_id = get_next_packet(f_in)
        except EOFError:
            break
        except Exception as e:
            print("Failed to get the next event ... Exception:", e)
            break

        if decode_all_data and data_id not in decoders:
            if data_id not in unrecognized_data_ids:
                unrecognized_data_ids.append(data_id)
            continue

        if data_id not in decoders: continue
        decoder = decoders[data_id]

        # Clear the tables if the next read could overflow them.
        # Only have to check this when the max table size is within
        # max_n_rows_per_packet of being full.
        if max_tbl_size + decoder.max_n_rows_per_packet() >= buffer_size:
            ch_groups = ch_groups_dict[id2dn_dict[data_id]]
            max_tbl_size = 0
            for group_info in ch_groups.values():
                tbl = group_info['table']
                if tbl.is_full():
                    group_path = group_info['group_path']
                    out_file = group_info['out_file']
                    lh5_store.write_object(tbl, group_path, out_file, n_rows=tbl.loc)
                    tbl.clear()
                if tbl.loc > max_tbl_size: max_tbl_size = tbl.loc
        else: max_tbl_size += decoder.max_n_rows_per_packet()

        tables = ch_tables_dict[data_id]
        decoder.decode_packet(packet, tables, packet_id, header_dict)

        if verbose:
            if n_max < np.inf and n_max > 0:
                update_len = 1
            else:
                update_len = f_in.tell() - file_position
                file_position = f_in.tell()
            update_progress(progress_bar, update_len)


    print("Done. Last packet ID:", packet_id)
    f_in.close()

    # final write to file
    for dec_name, ch_groups in ch_groups_dict.items():
        for group_info in ch_groups.values():
            tbl = group_info['table']
            if tbl.loc == 0: continue
            group_path = group_info['group_path']
            out_file = group_info['out_file']
            lh5_store.write_object(tbl, group_path, out_file, n_rows=tbl.loc)
            print('last write')
            tbl.clear()

    if len(unrecognized_data_ids) > 0:
        print("WARNING, Found the following unknown data IDs:")
        for data_id in unrecognized_data_ids:
            try:
                print("  {}: {}".format(data_id, id2dn_dict[data_id]))
            except KeyError:
                print("  {}: Unknown".format(data_id))
        print("hopefully they weren't important!\n")

    print("Wrote RAW File:\n    {}\nFILE INFO:".format(raw_file_pattern))
'''
