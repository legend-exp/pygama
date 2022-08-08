import numpy as np

import pygama.lgdo as lgdo


def test_datatype_name():
    array = lgdo.FixedSizeArray()
    assert array.datatype_name() == "fixedsize_array"


def test_form_datatype():
    array = lgdo.FixedSizeArray()
    assert array.form_datatype() == "fixedsize_array<0>{real}"


def test_init():
    attrs = {"attr1": 1}
    array = lgdo.FixedSizeArray(shape=(3,), dtype=np.float32, fill_val=42, attrs=attrs)
    assert (array.nda == np.full((3,), 42, np.float32)).all()
    assert array.attrs == attrs | {"datatype": "fixedsize_array<1>{real}"}
