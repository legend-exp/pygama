import numpy as np

from pygama.lgdo import Array, Table


def test_eval_dependency():
    obj = Table(
        col_dict={
            "a": Array(np.array([1, 2, 3, 4], dtype=np.float32)),
            "b": Array(np.array([5, 6, 7, 8], dtype=np.float32)),
        }
    )

    expr_config = {
        "O1": {"expression": "@p1 + @p2 * a**2", "parameters": {"p1": "2", "p2": "3"}},
        "O2": {"expression": "O1 - b"},
    }

    out_tbl = obj.eval(expr_config)
    assert list(out_tbl.keys()) == ["O1", "O2"]
    assert (out_tbl["O1"].nda == [5, 14, 29, 50]).all()
    assert (out_tbl["O2"].nda == [0, 8, 22, 42]).all()


def test_eval_math_functions():
    obj = Table(
        col_dict={
            "a": Array(np.array([1, 2, 3, 4], dtype=np.float32)),
            "b": Array(np.array([5, 6, 7, 8], dtype=np.float32)),
        }
    )

    expr_config = {
        "O1": {
            "expression": "exp(log(a))",
        }
    }

    out_tbl = obj.eval(expr_config)
    assert list(out_tbl.keys()) == ["O1"]
    assert (out_tbl["O1"].nda == [1, 2, 3, 4]).all()
