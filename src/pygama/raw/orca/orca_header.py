import json
import plistlib


class OrcaHeader(dict):
    """
    Orca file header object
    """
    def __init__(self, jsons=None, lgdo_scalar=None):
        if jsons is not None: self.update(json.loads(jsons))
        elif lgdo_scalar is not None: self.set_from_lgdo(lgdo_scalar)


    def set_from_lgdo(self, lgdo_scalar):
        if not isinstance(lgdo_scalar, lgdo.Scalar):
            print(f"Error: can't instantiate a header from a {type(lgdo_scalar)}.")
        else: self.update(json.loads(lgdo_scalar.value))


    def get_decoder_list(self):
        decoder_names = []
        dd = self['dataDescription']
        for class_key in dd.keys():
            for super_key in dd[class_key].keys():
                decoder_names.append(dd[class_key][super_key]['decoder'])
        return decoder_names


    def get_data_id(self, decoder_name):
        dd = self['dataDescription']
        for class_key in dd.keys():
            for super_key in dd[class_key].keys():
                if dd[class_key][super_key]['decoder'] == decoder_name:
                    return dd[class_key][super_key]['dataId']

    def get_id_to_object_name_dict(self, shift_data_id=True):
        id_dict = {}
        dd = self['dataDescription']
        for class_key in dd.keys():
            for super_key in dd[class_key].keys():
                data_id = dd[class_key][super_key]['dataId']
                if shift_data_id:
                    if data_id < 0: # short record
                        data_id = (-data_id) >> 26
                    else: data_id = data_id >> 18
                id_dict[data_id] = f'{class_key}:{super_key}'
        return id_dict

    def get_run_number(self):
        for d in self["ObjectInfo"]["DataChain"]:
            if "Run Control" in d: return d["Run Control"]["RunNumber"]
        raise ValueError("No run number found in header!")


    def get_object_info(self, orca_class_name):
        """
        returns a dict[crate#][card#] with all info from the header for each card with name
        orca_class_name.
        """
        object_info_dict = {}

        crates = self["ObjectInfo"]["Crates"]
        for crate in crates:
            object_info_dict[crate['CrateNumber']] = {}
            cards = crate["Cards"]
            for card in cards:
                if card["Class Name"] == orca_class_name:
                    object_info_dict[crate['CrateNumber']][card['Card']] = card

        if len(object_info_dict) == 0:
            raise KeyError(f'no object info for class {orca_class_name}')

        return object_info_dict


    def get_readout_info(self, orca_class_name, unique_id=-1):
        """
        returns a list with all the readout list info from the header with name
        orca_class_name.  optionally, if unique_id >= 0 only return the
        readout info for that Orca unique id number.
        """
        readout_info_list = []
        try:
            readouts = self["ReadoutDescription"]
            for readout in readouts:
                try:
                    if readout["name"] == orca_class_name:
                        if unique_id >= 0 and obj["uniqueID"] == unique_id: return readout
                        readout_info_list.append(readout)
                except KeyError: continue
        except KeyError: pass
        if len(readout_info_list) == 0:
            print('OrcaHeader::get_readout_info(): warning: no readout info '
              'for class name', orca_class_name)
        return readout_info_list


    def get_auxhw_info(self, orca_class_name, unique_id=-1):
        """
        returns a list with all the info from the AuxHw table of the header
        with name orca_class_name.  optionally, if unique_id >= 0 only return
        the object for that Orca unique id number.
        """
        auxhw_info_list = []
        try:
            objs = self["ObjectInfo"]["AuxHw"]
            for obj in objs:
                try:
                    if obj["Class Name"] == orca_class_name:
                        if unique_id >= 0 and obj["uniqueID"] == unique_id: return obj
                        auxhw_info_list.append(obj)
                except KeyError: continue
        except KeyError: pass
        if len(auxhw_info_list) == 0:
            print('OrcaHeader::get_auxhw_info(): warning: no object info '
                  'for class name', orca_class_name)
        return auxhw_info_list
