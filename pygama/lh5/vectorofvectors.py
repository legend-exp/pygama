import numpy as np
from .lh5 import get_lh5_datatype_name, get_lh5_element_type

class VectorOfVectors:
    """
    A variable-length array of variable-length arrays

    For now only a 1D vector of 1D vectors is supported. Internal representation
    is as two ndarrays, one to store the flattened data contiguosly and one to
    store the cumulative sum of lengths of each vector. 
    """ 
    def __init__(self, data_array=None, lensum_array=None, shape_guess=None, dtype=None, attrs={}):
        if lensum_array is None:
            self.lensum_array = Array(shape=(shape_guess[0],), dtype='uint32')
        else: self.lensum_array = lensum_array
        if data_array is None:
            length = np.prod(shape_guess)
            self.data_array = Array(shape=(length,), dtype=dtype)
        else: self.data_array = data_array
        self.dtype = self.data_array.dtype
        self.attrs = {}
        self.attrs.update(attrs)
        if 'datatype' in self.attrs:
            if self.attrs['datatype'] != self.form_datatype():
                print('VectorOfVectors: Warning: datatype does not match dtype!')
                print('datatype: ', self.attrs['datatype'])
                print('form_datatype(): ', self.form_datatype())
        else: self.attrs['datatype'] = self.form_datatype()


    def __len__(self):
        return len(self.lensum_array)


    def resize(self, new_size):
        self.lensum_array.resize(new_size)


    def form_datatype(self):
        et = get_lh5_element_type(self)
        return 'array<1>{array<1>{' + et + '}}'


    def set_vector(self, i_vec, nda):
        """Insert vector nda at location i_vec.

        self.data_array is doubled in length until nda can be appended to it.
        """
        if i_vec<0 or i_vec>len(self.lensum_array.nda)-1:
            print('VectorOfVectors: Error: bad i_vec', i_vec)
            return 
        if len(nda.shape) != 1:
            print('VectorOfVectors: Error: nda had bad shape', nda.shape)
            return
        start = 0 if i_vec == 0 else self.lensum_array.nda[i_vec-1]
        end = start + len(nda)
        while end >= len(self.data_array.nda):
            self.data_array.nda.resize(2*len(self.data_array.nda))
        self.data_array.nda[start:end] = nda
        self.lensum_array.nda[i_vec] = end
