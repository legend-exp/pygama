from .array import Array
from .lh5 import get_lh5_datatype_name, get_lh5_element_type

class ArrayOfEqualSizedArrays(Array):
    """
    An array of equal-sized arrays

    Arrays of equal size within a file but could be different from application
    to application. Canonical example: array of same-length waveforms.

    If shape is not "1D array of arrays of shape given by axes 1-N" (of nda)
    then specify the dimensionality split in the constructor.
    """
    def __init__(self, *args, dims=None, **kwargs):
        self.dims = dims
        super().__init__(*args, **kwargs)


    def __len__(self):
        return len(self.nda)


    def form_datatype(self):
        dt = get_lh5_datatype_name(self)
        nD = str(len(self.nda.shape))
        if self.dims is not None: nD = ','.join([str(i) for i in self.dims])
        et = get_lh5_element_type(self)
        return dt + '<' + nD + '>{' + et + '}'

