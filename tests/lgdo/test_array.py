import numpy as np

import pygama.lgdo as lgdo


def test_datatype_name():
    array = lgdo.Array()
    assert array.datatype_name() == "array"


def test_form_datatype():
    array = lgdo.Array(shape=(12, 34))
    assert array.form_datatype() == "array<2>{real}"


def test_init():
    attrs = {"attr1": 1}
    array = lgdo.Array(shape=(3,), dtype=np.float32, fill_val=42, attrs=attrs)
    assert (array.nda == np.full((3,), 42, np.float32)).all()
    assert array.attrs == attrs | {"datatype": "array<1>{real}"}


def test_resize():
    array = lgdo.Array(nda=np.array([1, 2, 3, 4]))
    array.resize(3)
    assert (array.nda == np.array([1, 2, 3])).all()
