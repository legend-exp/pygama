from .lh5 import get_lh5_datatype_name, get_lh5_element_type

class Array:
    """
    Holds an ndarray and attributes
    """
    def __init__(self, nda=None, shape=None, dtype=None, attrs={}):
        self.nda = nda if nda is not None else np.empty(shape, dtype=dtype)
        self.dtype = self.nda.dtype
        self.attrs = {}
        self.attrs.update(attrs)
        if 'datatype' in self.attrs:
            if self.attrs['datatype'] != self.form_datatype():
                print(type(self).__name__ + ': Warning: datatype does not match nda!')
                print('datatype: ', self.attrs['datatype'])
                print('form_datatype(): ', self.form_datatype())
                print('dtype:', self.dtype)
        else: self.attrs['datatype'] = self.form_datatype()


    def __len__(self):
        return len(self.nda)


    def resize(self, new_size):
        new_shape = (new_size,) + self.nda.shape[1:]
        self.nda.resize(new_shape)


    def form_datatype(self):
        dt = get_lh5_datatype_name(self)
        nD = str(len(self.nda.shape))
        et = get_lh5_element_type(self)
        return dt + '<' + nD + '>{' + et + '}'

