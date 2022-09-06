import numpy as np

from pygama.lgdo import Array, ArrayOfEqualSizedArrays,Table


def test_eval_dependency():
    obj = Table(
        col_dict={
            "a": Array(nda=np.array([1, 2, 3, 4], dtype=np.float32)),
            "b": Array(nda=np.array([5, 6, 7, 8], dtype=np.float32)),
            "c": ArrayOfEqualSizedArrays(nda=np.array([[1, 2, 3, 4],
                                                   [5, 6, 7, 8],
                                                   [9, 10, 11, 12],
                                                   [13, 14, 15, 16]], dtype=np.float32)),
            "d": ArrayOfEqualSizedArrays(nda=np.array([[21, 22, 23, 24],
                                                   [25, 26, 27, 8],
                                                   [29, 210, 211, 212],
                                                   [213, 214, 215, 216]], dtype=np.float32)),
        }
    )

    expr_config = {
        "O1": {"expression": "p1 + p2 * a**2", "parameters": {"p1": 2, "p2": 3}},
        "O2": {"expression": "O1 - b"},
        "O3": {"expression": "p1 + p2 * c", "parameters": {"p1": 2, "p2": 3}},
        "O4": {"expression": "O3 - d", "parameters": {"p1": 2, "p2": 3}},
        "O5": {"expression": "sum(c,axis=1)"},
        "O6": {"expression": "a>p1", "parameters": {"p1": 2}},
        "O7": {"expression": "c>p1", "parameters": {"p1": 2}}
    }

    out_tbl = obj.eval(expr_config)
    assert list(out_tbl.keys()) == ["O1", "O2","O3","O4", "O5","O6","O7"]
    assert (out_tbl["O1"].nda == [5, 14, 29, 50]).all()
    assert (out_tbl["O2"].nda == [0, 8, 22, 42]).all()
    assert (out_tbl["O3"].nda == [[ 5,  8, 11, 14],
                                           [17, 20, 23, 26],
                                           [29, 32, 35, 38],
                                           [41, 44, 47, 50]]).all()
    assert (out_tbl["O4"].nda == [[ -16.,  -14.,  -12.,  -10.],
                                           [  -8.,   -6.,   -4.,   18.],
                                           [   0., -178., -176., -174.],
                                           [-172., -170., -168., -166.]]).all()
    assert (out_tbl["O5"].nda == [10., 26., 42., 58.]).all()
    assert (out_tbl["O6"].nda == [False, False,  True,  True]).all()
    assert (out_tbl["O7"].nda == [[False, False,  True,  True],
                                           [ True,  True,  True,  True],
                                           [ True,  True,  True,  True],
                                           [ True,  True,  True,  True]]).all()


def test_eval_math_functions():
    obj = Table(
        col_dict={
            "a": Array(nda=np.array([1, 2, 3, 4], dtype=np.float32)),
            "b": Array(nda=np.array([5, 6, 7, 8], dtype=np.float32)),
            "c": ArrayOfEqualSizedArrays(nda=np.array([[1, 2, 3, 4],
                                                   [1, 2, 3, 4],
                                                   [1, 2, 3, 4],
                                                   [1, 2, 3, 4]], dtype=np.float32)),
        }
    )

    expr_config = {
        "O1": {"expression": "exp(log(a))"},
        "O2": {"expression": "exp(log(c))"}

    }

    out_tbl = obj.eval(expr_config)
    assert list(out_tbl.keys()) == ["O1","O2"]
    assert (out_tbl["O1"].nda == [1, 2, 3, 4]).all()
    assert (out_tbl["O2"].nda == np.array([[1, 2, 3, 4],
                                 [1, 2, 3, 4],
                                 [1, 2, 3, 4],
                                 [1, 2, 3, 4]])).all()
