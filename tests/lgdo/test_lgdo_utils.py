import numpy as np

import pygama.lgdo.lgdo_utils as lgdo_utils


def test_get_element_type():

    objs = [
        ('hi', 'string'),
        (True, 'bool'),
        (np.void(0), 'blob'),
        (int(0), 'real'),
        (np.uint8(0), 'real'),
        (float(0), 'real'),
        (1+1j, 'complex'),
        (b'hi', 'string'),
        (np.array(['hi']), 'string'),
    ]

    for obj, name in objs:
        get_name = lgdo_utils.get_element_type(obj)
        assert get_name == name


def test_parse_datatype():

    datatypes = [
        ('real', ('scalar', None, 'real')),
        ('array<1>{bool}', ('array', (1,), 'bool')),
        ('fixedsizearray<2>{real}', ('fixedsizearray', (2,), 'real')),
        ('arrayofequalsizedarrays<3,4>{complex}', ('arrayofequalsizedarrays', (3, 4), 'complex')),
        ('array<1>{array<1>{blob}}', ('array', (1,), 'array<1>{blob}')),
        ('struct{field1,field2,fieldn}', ('struct', None, ['field1', 'field2', 'fieldn'])),
        ('table{col1,col2,coln}', ('table', None, ['col1', 'col2', 'coln'])),
    ]

    for string, dt_tuple in datatypes:
        pd_dt_tuple = lgdo_utils.parse_datatype(string)
        assert pd_dt_tuple == dt_tuple
