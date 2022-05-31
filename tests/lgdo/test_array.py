import numpy as np
import pygama.lgdo as lgdo


def test_datatype_name():
    array = lgdo.Array()
    assert array.datatype_name() == 'array'


def test_form_datatype():
    array = lgdo.Array(shape=(12, 34))
    assert array.form_datatype() == 'array<2>{real}'


def test_init_value():
    array = lgdo.Array(shape=(3,), fill_val=42, dtype=np.float32)
    assert (array.nda == np.full((3,), 42, np.float32)).all()


def test_resize():
    array = lgdo.Array(nda=np.array([1, 2, 3, 4]))
    array.resize(3)
    assert (array.nda == np.array([1, 2, 3])).all()
