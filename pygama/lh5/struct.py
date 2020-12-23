from .lh5 import get_lh5_datatype_name, get_lh5_element_type

class Struct(dict):
    """
    A dictionary with an optional set of attributes.

    Don't allow to instantiate with a dictionary -- have to add fields
    one-by-one using add_field() to keep datatype updated
    """
    # TODO: overload setattr to require add_field for setting?
    def __init__(self, obj_dict={}, attrs={}):
        self.update(obj_dict)
        self.attrs = {}
        self.attrs.update(attrs)
        if 'datatype' in self.attrs:
            if self.attrs['datatype'] != self.form_datatype():
                print(type(self).__name__ + ': Warning: datatype does not match obj_dict!')
                print('datatype: ', self.attrs['datatype'])
                print('obj_dict.keys(): ', obj_dict.keys())
                print('form_datatype(): ', self.form_datatype())
        else: self.attrs['datatype'] = self.form_datatype()


    def add_field(self, name, obj):
        self[name] = obj
        self.attrs['datatype'] = self.form_datatype()


    def form_datatype(self):
        datatype = get_lh5_datatype_name(self)
        datatype += '{' + ','.join(self.keys()) + '}'
        return datatype

