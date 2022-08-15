import scipy

from pygama.pargen import energy_optimisation  # noqa: F401


def test_import():
    pass


def test_scipy_version():
    assert scipy.__version__ != ""
    assert scipy.__version__ is not None
