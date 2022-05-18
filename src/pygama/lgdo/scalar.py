import numpy as np

from .lgdo_utils import *


class Scalar:
    """
    Holds just a value and some attributes (datatype, units, ...)

    TODO: do scalars need proper numpy dtypes?
    """

    def __init__(self, value, attrs={}):
        """
        Parameters
        ----------
        value : scalar-like
            The value for this scalar
        attrs : dict (optional)
            A set of user attributes to be carried along with this lgdo
        """
        if not np.isscalar(value):
            print('Cannot instantiate a Scalar with a non-scalar value...')
            print('Setting value = 0')
            value = 0
        self.value = value
        self.attrs = dict(attrs)
        if 'datatype' in self.attrs:
            if self.attrs['datatype'] != self.form_datatype():
                print('Scalar: Warning: datatype does not match value!')
                print('datatype: ', self.attrs['datatype'])
                print('type(value): ', type(value).__name__)
        else: self.attrs['datatype'] = get_element_type(self.value)

    def datatype_name(self):
        """The name for this lgdo's datatype attribute"""
        if hasattr(self.value, 'datatype_name'): return self.value.datatype_name
        else: return get_element_type(self.value)

    def form_datatype(self):
        """Return this lgdo's datatype attribute string"""
        return self.datatype_name()

    def __str__(self):
        """Convert to string (e.g. for printing)"""
        string = str(self.value)
        tmp_attrs = self.attrs.copy()
        tmp_attrs.pop('datatype')
        if len(tmp_attrs) > 0: string += '\n' + str(tmp_attrs)
        return string

    def __repr__(self): return str(self)
