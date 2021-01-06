import numpy as np
from .lh5_utils import get_lh5_element_type

class Array:
    """
    Holds an ndarray and attributes
    """


    def __init__(self, nda=None, shape=None, dtype=None, attrs={}):
        """
        Parameters
        ----------

        nda : ndarray (optional)
            An ndarray to be used for this object's internal array. Note: the
            array is used directly, not copied. If not supplied, internal memory
            is newly allocated based on the shape and dtype arguments. 
        shape : tuple of ints (optional)
            A numpy-format shape specification for shape of the internal
            ndarray. Required if nda is None, otherwise unused.
        dtype : numpy dtype (optional)
            Specifies the type of the data in the array. Required if nda is
            None, otherwise unused.
        attrs : dict (optional)
            A set of user attributes to be carried along with this lh5 object
        """
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


    def dataype_name(self):
        """The name for this object's lh5 datatype attribute"""
        return 'array'


    def form_datatype(self):
        """Return this object's lh5 datatype attribute string"""
        dt = self.dataype_name()
        nD = str(len(self.nda.shape))
        et = get_lh5_element_type(self)
        return dt + '<' + nD + '>{' + et + '}'


    def __len__(self):
        """Provides __len__ for this array-like class"""
        return len(self.nda)


    def resize(self, new_size):
        """Resize the array to new_size (int)"""
        new_shape = (new_size,) + self.nda.shape[1:]
        self.nda.resize(new_shape)

