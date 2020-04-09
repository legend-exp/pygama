import sys
import numpy as np
import plistlib
from ..utils import update_progress
from .io_base import DataDecoder
from . import lh5

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
        """Overload to e.g. update decoded_values based on object_info.

        Otherwise just use header_dict and object_info as you need it
        """
        self.header_dict = header_dict
        self.object_info = get_object_info(header_dict, self.orca_class_name)

    def set_object_info(self, object_info):
        self.object_info = object_info

# NOTE: this import has to be after OrcaDecoder is defined.
# TODO: organize these classes better
from .orca_digitizers import *


def parse_header(xmlfile):
    """
    Opens the given file for binary read ('rb'), then grabs the first 8 bytes
    The first 4 bytes (1 long) of an orca data file are the total length in
    longs of the record
    The next 4 bytes (1 long) is the length of the header in bytes
    The header is then read in ...
    """
    with open(xmlfile, 'rb') as xmlfile_handle:
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

        #read in the next that-many bytes that occupy the plist header
        ba = bytearray(xmlfile_handle.read(j))

        #convert to string
        #the readPlistFromBytes method doesn't exist in 2.7
        if sys.version_info[0] < 3:
            header_string = ba.decode("utf-8")
            header_dict = plistlib.readPlistFromString(header_string)
        else:
            header_dict = plistlib.readPlistFromBytes(ba)
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
    returns a dict keyed by data id with all info from the header
    TODO: doesn't include all parts of the header yet!
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
        print('OrcaDecoder::get_object_info(): Warning: no object info')
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
    return (crate << 9) + ((card & 0xf) << 4) + (channel & 0xf)


def get_crate(ccc):
    return ccc >> 9


def get_card(ccc):
    return (ccc >> 4) & 0x1f


def get_channel(ccc):
    return ccc & 0xf


def process_orca(daq_filename, raw_filename, n_max, decoders, config, verbose, run=None, buffer_size=1024):
    """
    convert ORCA DAQ data to "raw" lh5
    """
    lh5_store = lh5.Store()

    f_in = open(daq_filename.encode('utf-8'), "rb")
    if f_in == None:
        print("Couldn't find the file %s" % daq_filename)
        sys.exit(0)

    # parse the header. save the length so we can jump past it later
    reclen, reclen2, header_dict = parse_header(daq_filename)

    # figure out the total size
    SEEK_END = 2
    f_in.seek(0, SEEK_END)
    file_size = float(f_in.tell())
    f_in.seek(0, 0)  # rewind
    file_size_MB = file_size / 1e6
    print("Total file size: {:.3f} MB".format(file_size_MB))

    if run is not None:
        run = get_run_number(header_dict)
    print("Run number:", run)

    # figure out which decoders we can use and build a dictionary from
    # dataID to decoder
    decoders = {}
    id2dn_dict = get_id_to_decoder_name_dict(header_dict)
    if verbose:
        print("Data IDs present in ORCA file header are:")
        for data_id in id2dn_dict:
            print(f"    {data_id}: {id2dn_dict[data_id]}")
    dn2id_dict = {name:data_id for data_id, name in id2dn_dict.items()}
    for sub in OrcaDecoder.__subclasses__():
        decoder = sub() # instantiate the class
        if decoder.decoder_name in dn2id_dict:
            # Later: allow to turn on / off decoders in exp.json
            if decoder.decoder_name != 'ORSIS3302DecoderForEnergy': continue
            decoder.dataID = dn2id_dict[decoder.decoder_name]
            decoder.set_object_info(get_object_info(header_dict, decoder.orca_class_name))
            decoders[decoder.dataID] = decoder
    if verbose:
        print("pygama will run these decoders:")
        for data_id, dec in decoders.items():
            print("   ", dec.decoder_name+ ", id =", data_id)

    # Set up dataframe buffers -- for now, one for each decoder
    # Later: control in intercom
    tbs = {}
    for data_id, dec in decoders.items():
        tbs[data_id] = lh5.Table(buffer_size)
        dec.initialize_lh5_table(tbs[data_id])

    # -- scan over raw data --
    print("Beginning daq-to-raw processing ...")

    packet_id = 0  # number of events decoded
    unrecognized_data_ids = []

    # skip the header using reclen from before
    # reclen is in number of longs, and we want to skip a number of bytes
    f_in.seek(reclen * 4)

    # start scanning
    while (packet_id < n_max and f_in.tell() < file_size):
        packet_id += 1

        if verbose and packet_id % 1000 == 0:
            update_progress(float(f_in.tell()) / file_size)

        try:
            packet, data_id = get_next_packet(f_in)
        except EOFError:
            break
        except Exception as e:
            print("Failed to get the next event ... Exception:", e)
            break

        if data_id not in decoders:
            if data_id not in unrecognized_data_ids:
                unrecognized_data_ids.append(data_id)
            continue

        if data_id in tbs.keys():
            tb = tbs[data_id]
            decoder.decode_packet(packet, tb, packet_id, header_dict)
            # write to the output file when a buffer gets full
            if tb.is_full():
                lh5_store.write_object(tb, id2dn_dict[data_id], raw_filename, n_rows=tb.size)
                tb.clear()

    print("Done. Last packet ID:", packet_id)
    f_in.close()

    # final write to file
    for data_id in tbs.keys():
        tb = tbs[data_id]
        if tb.loc == 0: continue
        lh5_store.write_object(tb, id2dn_dict[data_id], raw_filename, n_rows=tb.loc)
        tb.clear()

    if verbose:
        update_progress(1)

    if len(unrecognized_data_ids) > 0:
        print("WARNING, Found the following unknown data IDs:")
        for data_id in unrecognized_data_ids:
            print("  {}: {}".format(data_id, id2dn_dict[data_id]))
        print("hopefully they weren't important!\n")

    print("Wrote RAW File:\n    {}\nFILE INFO:".format(raw_filename))
