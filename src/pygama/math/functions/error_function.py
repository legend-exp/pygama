import math

import numba as nb
import numpy as np


@nb.vectorize([nb.float32(nb.float32),
nb.float64(nb.float64)])
def nb_erf(x: float) -> float:
    """
    Numba version of error function.
    Vectorization is necessary here for the math.erf
    This runs faster than numpy vectorized and the
    out-of-the-box math.erf

    Parameters
    ----------
    x : float or array-like
        The input data

    Returns
    -------
    math.erf(x): float or array-like
        Error function acting on the input

    """
    return math.erf(x)



@nb.vectorize([nb.float32(nb.float32),
nb.float64(nb.float64)])
def nb_erfc(x:float) -> float:
    """
    Numba version of complementary error function
    Vectorization is necessary here for the math.erfc
    This runs faster than numpy vectorized and the
    out-of-the-box math.erfc

    Parameters
    ----------
    x : float or array-like
        The input data

    Returns
    -------
    math.erfc(x): float or array-like
        Complementary error function acting on the input
    """
    return math.erfc(x)
