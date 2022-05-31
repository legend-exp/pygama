import pygama.lgdo as lgdo


def test_datatype_name():
    scalar = lgdo.Scalar(value=42)
    assert scalar.datatype_name() == 'real'


def test_form_datatype():
    scalar = lgdo.Scalar(value=42)
    assert scalar.form_datatype() == 'real'


def test_init_value():
    scalar = lgdo.Scalar(value=42)
    assert scalar.value == 42
