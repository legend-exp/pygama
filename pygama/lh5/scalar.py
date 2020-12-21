from .lh5 import get_lh5_datatype_name, get_lh5_element_type

class Scalar:
    """
    Holds just a value and some attributes (datatype, units, ...)
    """
    def __init__(self, value, attrs={}):
        self.value = value
        self.attrs = {}
        self.attrs.update(attrs)
        if 'datatype' in self.attrs:
            if self.attrs['datatype'] != get_lh5_element_type(self.value):
                print('Scalar: Warning: datatype does not match value!')
                print('datatype: ', self.attrs['datatype'])
                print('type(value): ', type(value).__name__)
        else: self.attrs['datatype'] = get_lh5_element_type(self.value)

