import numpy as np

import pygama.lgdo as lgdo


def test_datatype_name():
    aoesa = lgdo.ArrayOfEqualSizedArrays()
    assert aoesa.datatype_name() == "array_of_equalsized_arrays"


def test_form_datatype():
    aoesa = lgdo.ArrayOfEqualSizedArrays(dims=(2, 3))
    assert aoesa.form_datatype() == "array_of_equalsized_arrays<2,3>{real}"


def test_init():
    attrs = {"attr1": 1}
    aoesa = lgdo.ArrayOfEqualSizedArrays(
        dims=(2, 3), dtype=np.float32, fill_val=42, attrs=attrs
    )
    assert aoesa.dims == (2, 3)
    assert (aoesa.nda == np.full((2, 3), 42, np.float32)).all()
    assert aoesa.attrs == attrs | {"datatype": "array_of_equalsized_arrays<2,3>{real}"}
