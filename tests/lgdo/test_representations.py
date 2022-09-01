import numpy as np

from pygama import lgdo


def test_representations():
    objs = {
        "Scalar": lgdo.Scalar("test", attrs={"unit": "ns"}),
        "Array1": lgdo.Array(np.random.rand(3), attrs={"unit": "ns"}),
        "Array2": lgdo.Array(np.random.rand(10), attrs={"unit": "ns"}),
        "Array3": lgdo.Array(np.random.rand(100), attrs={"unit": "ns"}),
        "ArrayOfEqualSizedArrays": lgdo.ArrayOfEqualSizedArrays(
            nda=np.random.rand(10, 100), attrs={"unit": "ns"}
        ),
        "Struct": lgdo.Struct(
            {
                "first": lgdo.Array(np.random.rand(100), attrs={"unit": "ns"}),
                "second": lgdo.Scalar(3.45, attrs={"unit": "ns"}),
                "third": lgdo.Array(np.random.rand(3), attrs={"unit": "ns"}),
                "fourth": lgdo.ArrayOfEqualSizedArrays(
                    nda=np.random.rand(10, 100), attrs={"unit": "ns"}
                ),
            },
            attrs={"unit": "ns"},
        ),
        "VectorOfVectors": lgdo.VectorOfVectors(
            flattened_data=lgdo.Array(np.random.rand(1000)),
            cumulative_length=lgdo.Array(np.array([5, 12, 34, 49, 150])),
            attrs={"unit": "ns"},
        ),
        "Table": lgdo.Table(
            col_dict={
                "first": lgdo.Array(np.random.rand(100), attrs={"unit": "ns"}),
                "second": lgdo.Array(np.random.rand(100)),
                "third": lgdo.Array(np.random.rand(100), attrs={"unit": "ns"}),
            },
            attrs={"greeting": "ciao"},
        ),
        "WaveformTable1": lgdo.WaveformTable(
            values=lgdo.VectorOfVectors(
                flattened_data=lgdo.Array(np.random.rand(1000)),
                cumulative_length=lgdo.Array(np.array([5, 12, 74, 230])),
            ),
            attrs={"greeting": "ciao"},
        ),
        "WaveformTable2": lgdo.WaveformTable(
            values=lgdo.ArrayOfEqualSizedArrays(nda=np.random.rand(10, 1000)),
            attrs={"greeting": "ciao"},
        ),
        "WaveformTable3": lgdo.WaveformTable(
            values=lgdo.ArrayOfEqualSizedArrays(nda=np.random.rand(10, 100)),
            attrs={"greeting": "ciao"},
        ),
    }

    for k, it in objs.items():
        print(f">>> {k}")  # noqa: T201
        print(repr(it))  # noqa: T201
        print()  # noqa: T201
        print(f">>> print({k})")  # noqa: T201
        print(it)  # noqa: T201
        print()  # noqa: T201
