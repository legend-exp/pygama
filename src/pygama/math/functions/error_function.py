import math

import numba as nb
import numpy as np

kwd = {"parallel": False, "fastmath": True}


@nb.njit(**kwd)
def nb_erf(x):
    """
    Numba version of error function
    """
    y = np.empty_like(x)
    for i in nb.prange(len(x)):
        y[i] = math.erf(x[i])
    return y


@nb.njit(**kwd)
def nb_erfc(x):
    """
    Numba version of complementary error function
    """
    y = np.empty_like(x)
    for i in nb.prange(len(x)):
        y[i] = math.erfc(x[i])
    return y
