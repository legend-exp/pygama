import numpy as np
import pandas as pd
import pytest

import pygama.lgdo as lgdo
from pygama.lgdo.table import Table


def test_init():
    tbl = Table()
    assert tbl.size == 1024
    assert tbl.loc == 0

    tbl = Table(size=10)
    assert tbl.size == 10

    col_dict = {
        "a": lgdo.Array(nda=np.array([1, 2, 3, 4])),
        "b": lgdo.Array(nda=np.array([5, 6, 7, 8])),
    }

    tbl = Table(col_dict=col_dict)
    assert tbl.size == 4

    tbl = Table(size=3, col_dict=col_dict)
    assert tbl.size == 3


def test_datatype_name():
    tbl = Table()
    assert tbl.datatype_name() == "table"


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
    tbl.add_field("a", lgdo.Array(np.array([1, 2, 3])), use_obj_size=True)
    assert tbl.size == 3

    with pytest.raises(TypeError):
        tbl.add_field("s", lgdo.Scalar(value=69))


def test_add_column():
    tbl = Table()
    tbl.add_column("a", lgdo.Array(np.array([1, 2, 3])), use_obj_size=True)
    assert tbl.size == 3


def test_join():
    tbl1 = Table(size=3)
    tbl1.add_field("a", lgdo.FixedSizeArray(np.array([1, 2, 3])))
    tbl1.add_field("b", lgdo.Array(np.array([1, 2, 3])))
    assert list(tbl1.keys()) == ["a", "b"]

    tbl2 = Table(size=3)
    tbl2.add_field("c", lgdo.Array(np.array([4, 5, 6])))
    tbl2.add_field("d", lgdo.Array(np.array([9, 9, 10])))

    tbl1.join(tbl2)
    assert list(tbl1.keys()) == ["a", "b", "c", "d"]

    tbl2.join(tbl1, cols=("a"))
    assert list(tbl2.keys()) == ["c", "d", "a"]


def test_get_dataframe():
    tbl = Table(3)
    tbl.add_column("a", lgdo.Array(np.array([1, 2, 3])))
    tbl.add_column("b", lgdo.Array(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])))
    tbl.add_column(
        "c",
        lgdo.VectorOfVectors(
            flattened_data=lgdo.Array(np.array([0, 1, 2, 3, 4, 5, 6])),
            cumulative_length=lgdo.Array(np.array([3, 4, 7])),
        ),
    )
    df = tbl.get_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert list(df.keys()) == ["a", "b", "c"]


def test_remove_column():
    col_dict = {
        "a": lgdo.Array(nda=np.array([1, 2, 3, 4])),
        "b": lgdo.Array(nda=np.array([5, 6, 7, 8])),
        "c": lgdo.Array(nda=np.array([9, 10, 11, 12])),
    }

    tbl = Table(col_dict=col_dict)

    tbl.remove_column("a")
    assert list(tbl.keys()) == ["b", "c"]

    tbl.remove_column("c")
    assert list(tbl.keys()) == ["b"]
