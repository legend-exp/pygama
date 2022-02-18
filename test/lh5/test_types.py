import pygama.lh5 as lh5


def test_array():
    a = lh5.Array(shape=(1), dtype=float)
    assert a.dataype_name() == 'array'
