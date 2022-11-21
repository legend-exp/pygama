import pygama.lgdo as lgdo


def test_datatype_name():
    struct = lgdo.Struct()
    assert struct.datatype_name() == "struct"


def test_form_datatype():
    struct = lgdo.Struct()
    assert struct.form_datatype() == "struct{}"


def test_update_datatype():
    struct = lgdo.Struct()
    struct.update_datatype()
    assert struct.attrs["datatype"] == "struct{}"


def test_init():
    obj_dict = {"scalar1": lgdo.Scalar(value=10)}
    attrs = {"attr1": 1}
    struct = lgdo.Struct(obj_dict=obj_dict, attrs=attrs)
    assert dict(struct) == obj_dict
    assert struct.attrs == attrs | {"datatype": "struct{scalar1}"}


def test_add_field():
    struct = lgdo.Struct()
    struct.add_field("scalar1", lgdo.Scalar(value=10))

    assert struct.attrs["datatype"] == "struct{scalar1}"
    assert struct["scalar1"].__class__.__name__ == "Scalar"

    struct.add_field("array1", lgdo.Array(shape=(700, 21), dtype="f", fill_val=2))
    assert struct.attrs["datatype"] == "struct{scalar1,array1}"


def test_remove_field():
    struct = lgdo.Struct()
    struct.add_field("scalar1", lgdo.Scalar(value=10))
    struct.add_field("array1", lgdo.Array(shape=(10), fill_val=0))

    struct.remove_field("scalar1")
    assert list(struct.keys()) == ["array1"]

    struct.remove_field("array1", delete=True)
    assert list(struct.keys()) == []
