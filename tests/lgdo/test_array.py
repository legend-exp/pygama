import numpy as np

from pygama import lgdo
from pygama.lgdo import Array


def test_datatype_name():
    array = Array()
    assert array.datatype_name() == "array"


def test_form_datatype():
    array = Array(shape=(12, 34))
    assert array.form_datatype() == "array<2>{real}"


def test_init():
    attrs = {"attr1": 1}
    array = Array(shape=(3,), dtype=np.float32, fill_val=42, attrs=attrs)
    assert (array.nda == np.full((3,), 42, np.float32)).all()
    assert array.attrs == attrs | {"datatype": "array<1>{real}"}


def test_resize():
    array = Array(nda=np.array([1, 2, 3, 4]))
    array.resize(3)
    assert (array.nda == np.array([1, 2, 3])).all()


def test_copy():
    a1 = Array(np.array([1, 2, 3, 4]))
    a2 = lgdo.copy(a1)
    assert a1 == a2


def test_insert():
    a = Array(np.array([1, 2, 3, 4]))
    a.insert(2, [-1, -1])
    assert a == Array([1, 2, -1, -1, 3, 4])
