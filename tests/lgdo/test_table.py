import numpy as np
from pygama.lgdo.table import Table
import pygama.lgdo as lgdo


def test_init():
    tbl = Table()
    assert tbl.size == 1024
    assert tbl.loc == 0

    tbl = Table(size=10)
    assert tbl.size == 10

    col_dict = {
        'a': lgdo.Array(nda=np.array([1, 2, 3, 4])),
        'b': lgdo.Array(nda=np.array([5, 6, 7, 8]))
    }

    tbl = Table(col_dict=col_dict)
    assert tbl.size == 4

    tbl = Table(size=3, col_dict=col_dict)
    assert tbl.size == 3


def test_datatype_name():
    tbl = Table()
    assert tbl.datatype_name() == 'table'


def test_push_row():
    tbl = Table()
    tbl.push_row()
    assert tbl.loc == 1


def test_is_full():
    tbl = Table(size=2)
    tbl.push_row()
    assert tbl.is_full() is False
    tbl.push_row()
    assert tbl.is_full() is True


def test_clear():
    tbl = Table()
    tbl.push_row()
    tbl.clear()
    assert tbl.loc == 0


def test_add_field():
    tbl = Table()
    tbl.add_field('a', lgdo.Array(np.array([1, 2, 3])), use_obj_size=True)
    assert tbl.size == 3


def test_add_column():
    tbl = Table()
    tbl.add_column('a', lgdo.Array(np.array([1, 2, 3])), use_obj_size=True)
    assert tbl.size == 3


def test_join():
    tbl1 = Table()
    tbl1.add_field('a', lgdo.Array(np.array([1, 2, 3])))

    tbl2 = Table()
    tbl2.add_field('b', lgdo.Array(np.array([4, 5, 6, 7])))
    tbl2.add_field('c', lgdo.Array(np.array([9, 9, 10, 11])))

    tbl1.join(tbl2)
    assert list(tbl1.keys()) == ['a', 'b', 'c']
