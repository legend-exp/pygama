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
