import numpy as np

from .lgdo_utils import *


class Array:
    """Holds an ndarray and attributes

    Array (and the other various array types) holds an "nda" instead of deriving
    from ndarray for the following reasons:

    - it keeps management of the nda totally under the control of the user. The
      user can point it to another object's buffer, grab the nda and toss the
      Array, etc.
    - it allows the management code to send just the nda's the central routines
      for data manpulation. Keeping lgdo's out of that code allows for more
      standard, reusable, and (we expect) performant python
    - it allows the first axis of the nda to be treated as "special" for storage
      in Tables

    """


    def __init__(self, nda=None, shape=(), dtype=None, fill_val=None, attrs={}):
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
        fill_val : scalar or None
            If None, memory is allocated without initialization. Otherwise, the
            array is allocated with all elements set to the corresponding fill
            value. If nda is not None, this parameter is ignored
        attrs : dict (optional)
            A set of user attributes to be carried along with this lgdo
        """
        if nda is None:
            if fill_val is None: nda = np.empty(shape, dtype=dtype)
            elif fill_val == 0: nda = np.zeros(shape, dtype=dtype)
            else: nda = np.full(shape, fill_val, dtype=dtype)
        self.nda = nda
        self.dtype = self.nda.dtype
        self.attrs = dict(attrs)
        if 'datatype' in self.attrs:
            if self.attrs['datatype'] != self.form_datatype():
                print(type(self).__name__ + ': Warning: datatype does not match nda!')
                print('datatype: ', self.attrs['datatype'])
                print('form_datatype(): ', self.form_datatype())
                print('dtype:', self.dtype)
        else: self.attrs['datatype'] = self.form_datatype()


    def datatype_name(self):
        """The name for this lgdo's datatype attribute"""
        return 'array'


    def form_datatype(self):
        """Return this lgdo's datatype attribute string"""
        dt = self.datatype_name()
        nD = str(len(self.nda.shape))
        et = get_element_type(self)
        return dt + '<' + nD + '>{' + et + '}'


    def __len__(self):
        """Provides __len__ for this array-like class"""
        return len(self.nda)


    def resize(self, new_size):
        """Resize the array to new_size (int)"""
        new_shape = (new_size,) + self.nda.shape[1:]
        self.nda.resize(new_shape, refcheck=True)

    def __str__(self):
        """Convert to string (e.g. for printing)"""
        tmp_attrs = self.attrs.copy()
        datatype = tmp_attrs.pop('datatype')
        string = datatype + " = " + str(self.nda)
        if len(tmp_attrs) > 0: string += '\n attrs = ' + str(tmp_attrs)
        return string

    def __repr__(self): return str(self)
