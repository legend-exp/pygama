import numpy as np
from .lh5_utils import get_lh5_element_type
from .array import Array

class VectorOfVectors:
    """
    A variable-length array of variable-length arrays

    For now only a 1D vector of 1D vectors is supported. Internal representation
    is as two ndarrays, one to store the flattened data contiguosly and one to
    store the cumulative sum of lengths of each vector. 
    """ 


    def __init__(self, flattened_data=None, cumulative_length=None, shape_guess=None, dtype=None, attrs={}):
        """
        Parameters
        ----------
        flattened_data : lh5.Array (optional)
            If not None, used as the internal memory array for flattened_data.
            Otherwise, an internal flattened_data is allocated based on
            shape_guess and dtype
        cumulative_length : lh5.Array (optional)
            If not None, used as the internal memory array for
            cumulative_length. Should be dtype uint32. If cumulative_length is
            None, an internal cumulative_length is allocated based on the first
            element of shape_guess
        shape_guess : 2-tuple of ints (optional)
            A numpy-format shape specification, required if either of
            flattened_data or cumulative_length are not supplied.
            The first element should not be a guess and sets the number of
            vectors to be stored. The second element is a guess or approximation
            of the typical length of a stored vector, used to set the initial
            length of flattened_data if it was not supplied.
        dtype : numpy dtype
            Sets the type of data stored in flattened_data. Required if
            flattened_data is None
        attrs : dict (optional)
            A set of user attributes to be carried along with this lh5 object
        """
        if cumulative_length is None:
            self.cumulative_length = Array(shape=(shape_guess[0],), dtype='uint32')
        else: self.cumulative_length = cumulative_length
        if flattened_data is None:
            length = np.prod(shape_guess)
            self.flattened_data = Array(shape=(length,), dtype=dtype)
        else: self.flattened_data = flattened_data
        self.dtype = self.flattened_data.dtype
        self.attrs = {}
        self.attrs.update(attrs)
        if 'datatype' in self.attrs:
            if self.attrs['datatype'] != self.form_datatype():
                print('VectorOfVectors: Warning: datatype does not match dtype!')
                print('datatype: ', self.attrs['datatype'])
                print('form_datatype(): ', self.form_datatype())
        else: self.attrs['datatype'] = self.form_datatype()


    def datatype_name(self):
        """The name for this object's lh5 datatype attribute"""
        return 'array'


    def __len__(self):
        """Provides __len__ for this array-like class"""
        return len(self.cumulative_length)


    def resize(self, new_size):
        self.cumulative_length.resize(new_size)


    def form_datatype(self):
        """Return this object's lh5 datatype attribute string"""
        et = get_lh5_element_type(self)
        return 'array<1>{array<1>{' + et + '}}'


    def set_vector(self, i_vec, nda):
        """Insert vector nda at location i_vec.

        self.flattened_data is doubled in length until nda can be appended to it.
        """
        if i_vec<0 or i_vec>len(self.cumulative_length.nda)-1:
            print('VectorOfVectors: Error: bad i_vec', i_vec)
            return 
        if len(nda.shape) != 1:
            print('VectorOfVectors: Error: nda had bad shape', nda.shape)
            return
        start = 0 if i_vec == 0 else self.cumulative_length.nda[i_vec-1]
        end = start + len(nda)
        while end >= len(self.flattened_data.nda):
            self.flattened_data.nda.resize(2*len(self.flattened_data.nda))
        self.flattened_data.nda[start:end] = nda
        self.cumulative_length.nda[i_vec] = end
