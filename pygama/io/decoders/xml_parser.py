import plistlib
import sys
import pandas as pd


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


def get_data_id(headerDict, class_name, super_name):
    """
    stored like this:
    `headerDict["dataDescription"]["ORRunModel"]["Run"]["dataId"]`
    but integer needs to be bitshifted by 18
    """
    id_int = headerDict["dataDescription"][class_name][super_name]["dataId"]

    return id_int >> 18


def flip_data_ids(headerDict):
    """
    Returns an inverted dictionary such that:
    Could be extended somehow to give you all the supers associated with a given class name (maybe like)
    flipped[dataId] = [class_key, [super1, super2, ...]]
    """
    flipped = dict()
    # headerDict["dataDescription"][class_name][super_name]["dataId"]
    for class_key in headerDict["dataDescription"].keys():
        super_keys_list = []
        for super_key in headerDict["dataDescription"][class_key].keys():
            super_keys_list.append(super_key)
            ID_val = (headerDict["dataDescription"][class_key][super_key]
                      ["dataId"]) >> 18
            flipped[ID_val] = [class_key, super_keys_list]

    # this one just gives a single super             flipped[dataId] = [class_key, super_key]
    # for class_key in headerDict["dataDescription"].keys():
    #     super_keys_list = headerDict["dataDescription"][class_key].keys()
    #     ID_val = (headerDict["dataDescription"][class_key][super_keys_list[0]]["dataId"])>>18
    #     flipped[ID_val] = [class_key,super_keys_list]

    return flipped


def get_decoder_for_id(headerDict):
    """
    Returns a dictionary that goes:
    `dict[dataIdNum] = "decoderName"`
    e.g: d[5] = 'ORSIS3302DecoderForEnergy'
    """
    d = dict()
    for class_key in headerDict["dataDescription"].keys():
        super_keys_list = []
        for super_key in headerDict["dataDescription"][class_key].keys():
            super_keys_list.append(super_key)
            ID_val = (headerDict["dataDescription"][class_key][super_key]
                      ["dataId"]) >> 18
            decoderName = headerDict["dataDescription"][class_key][super_key][
                "decoder"]

            d[ID_val] = decoderName

    return d


def get_object_info(headerDict, class_name):
    """
    returns a dict keyed by data id with all info from the header
    TODO: doesn't include all parts of the header yet!
    """
    object_info_list = []

    crates = headerDict["ObjectInfo"]["Crates"]
    for crate in crates:
        cards = crate["Cards"]
        for card in cards:
            if card["Class Name"] == class_name:
                card["Crate"] = crate["CrateNumber"]
                object_info_list.append(card)

    # AuxHw = headerDict["ObjectInfo"]["AuxHw"]
    # print("AUX IS:")
    # for aux in AuxHw:
    # print(aux.keys())
    # exit()

    if len(object_info_list) == 0:
        # print("Warning: no object info parsed for {}".format(class_name))
        return None
    df = pd.DataFrame.from_dict(object_info_list)
    df.set_index(['Crate', 'Card'], inplace=True)
    return df
