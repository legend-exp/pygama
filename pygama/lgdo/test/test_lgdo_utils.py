import pytest
import pygama.lgdo.lgdo_utils as lgdo_utils
import numpy as np
from numpy.testing import assert_

objs = [ ('hi', 'string'),
         (True, 'bool'),
         (np.void(0), 'blob'),
         (np.int(0), 'real'),
         (np.uint8(0), 'real'),
         (np.float(0), 'real'),
         (1+1j, 'complex'),
         (b'hi', 'string'),
         (np.array(['hi']), 'string'),
       ]

def test_get_element_type():
    for obj, name in objs:
        get_name = lgdo_utils.get_element_type(obj) 
        assert_(get_name == name, f'error with {name}: got {get_name}')

