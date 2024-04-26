from math import erf, erfc
from typing import Union

import numba as nb
import numpy as np


@nb.vectorize([nb.float32(nb.float32), nb.float64(nb.float64)])
def nb_erf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    r"""
    Numba version of error function.
    Vectorization is necessary here for the math.erf
    This runs faster than numpy vectorized and the
    out-of-the-box math.erf

    Parameters
    ----------
    x
        The input data

    Returns
    -------
    math.erf(x)
        Error function acting on the input
    """

    return erf(x)


@nb.vectorize([nb.float32(nb.float32), nb.float64(nb.float64)])
def nb_erfc(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    r"""
    Numba version of complementary error function
    Vectorization is necessary here for the math.erfc
    This runs faster than numpy vectorized and the
    out-of-the-box math.erfc

    Parameters
    ----------
    x
        The input data

    Returns
    -------
    math.erfc(x)
        Complementary error function acting on the input
    """

    return erfc(x)
