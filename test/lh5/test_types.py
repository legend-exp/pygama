import pygama.lgdo as lgdo


def test_array():
    a = lgdo.Array(shape=(1), dtype=float)
    assert a.dataype_name() == 'array'
