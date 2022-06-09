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


    def __init__(self, dims=None, nda=None, shape=(), dtype=None, fill_val=None, attrs={}):
        """
        Parameters
        ----------
        dims : tuple of ints (optional)
            specifies the dimensions required for building the
            ArrayOfEqualSizedArrays' datatype attribute
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
        fill_val : scalar or None
            If None, memory is allocated without initialization. Otherwise, the
            array is allocated with all elements set to the corresponding fill
            value. If nda is not None, this parameter is ignored
        attrs : dict (optional)
            A set of user attributes to be carried along with this lgdo

        See Also
        --------
        :class:`.Array`
        """
        self.dims = dims
        super().__init__(nda=nda, shape=shape, dtype=dtype, fill_val=fill_val, attrs=attrs)


    def datatype_name(self):
        """The name for this lgdo's datatype attribute"""
        return 'array_of_equalsized_arrays'


    def form_datatype(self):
        """Return this lgdo's datatype attribute string"""
        dt = self.datatype_name()
        nD = str(len(self.nda.shape))
        if self.dims is not None: nD = ','.join([str(i) for i in self.dims])
        et = get_element_type(self)
        return dt + '<' + nD + '>{' + et + '}'


    def __len__(self):
        """Provides __len__ for this array-like class"""
        return len(self.nda)
