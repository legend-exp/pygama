from .array import Array
from .lgdo_utils import *


class ArrayOfEqualSizedArrays(Array):
    """
    An array of equal-sized arrays

    Arrays of equal size within a file but could be different from application
    to application. Canonical example: array of same-length waveforms.

    If shape is not "1D array of arrays of shape given by axes 1-N" (of nda)
    then specify the dimensionality split in the constructor.
    """


    def __init__(self, dims=None, nda=None, shape=None, dtype=None, fill_val=None, attrs={}):
        """
        Parameters
        ----------
        dims : tuple of ints (optional)
            specifies the dimensions required for building the
            ArrayOfEqualSizedArrays' datatype attribute

        See Array.__init__ for optional args
        """
        self.dims = dims
        super().__init__(nda=nda, shape=shape, dtype=dtype, fill_val=fill_val, attrs=attrs)


    def dataype_name(self):
        """The name for this lgdo's datatype attribute"""
        return 'array_of_equalsized_arrays'


    def form_datatype(self):
        """Return this lgdo's datatype attribute string"""
        dt = self.dataype_name()
        nD = str(len(self.nda.shape))
        if self.dims is not None: nD = ','.join([str(i) for i in self.dims])
        et = get_element_type(self)
        return dt + '<' + nD + '>{' + et + '}'


    def __len__(self):
        """Provides __len__ for this array-like class"""
        return len(self.nda)