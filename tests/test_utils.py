import pygama.utils as pgu


def test_math_numba_defaults():
    assert pgu.numba_math_defaults_kwargs.fastmath
    assert not pgu.numba_math_defaults_kwargs.parallel

    pgu.numba_math_defaults.fastmath = False
    assert not pgu.numba_math_defaults.fastmath
