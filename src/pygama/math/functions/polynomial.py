import numba as nb
import numpy as np

from pygama.utils import numba_math_defaults as nb_defaults


@nb.njit(**nb_defaults(parallel=False))
def nb_poly(x: np.ndarray, pars: np.ndarray) -> np.ndarray:
    r"""
    A polynomial function with pars following the numpy polynomial convention. It computes:


    .. math::
        y = a_n x^n + ... + a_1 x + a_0


    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    Parameters
    ----------
    x
        Input data
    pars
        Coefficients of the polynomial, in  numpy polynomial convention

    Returns
    -------
    result
        The polynomial defined by pars at the evaluated at x

    Notes
    -----
    This follows the :func:`numpy.polynomial` convention of :math:`a_0 + a_1 x +.... + a_n x^n`
    """

    result = x * 0  # do x*0 to keep shape of x (scalar or array)
    if len(pars) == 0:
        return result
    result += pars[0]
    y = x
    for i in nb.prange(1, len(pars)):
        result += pars[i] * x
        x = x * y
    return result
