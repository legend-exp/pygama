import pygama.lgdo as lgdo


def test_datatype_name():
    struct = lgdo.Struct()
    assert struct.datatype_name() == 'struct'


def test_form_datatype():
    struct = lgdo.Struct()
    assert struct.form_datatype() == 'struct{}'


def test_add_field():
    # set up, add scalar object
    struct = lgdo.Struct()
    struct.add_field('scalar1', lgdo.Scalar(value=10))

    # verify 'struct{scalar1}' is in attributes
    assert struct.attrs['datatype'] == 'struct{scalar1}'

    # and the correct type
    assert struct['scalar1'].__class__.__name__ == 'Scalar'

    # add array and test updated attributes
    struct.add_field('array1', lgdo.Array(shape=(700, 21), dtype='f', fill_val=2))
    assert struct.attrs['datatype'] == 'struct{scalar1,array1}'
