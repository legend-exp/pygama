import math

import numba as nb
import numpy as np


@nb.vectorize([nb.float32(nb.float32),
nb.float64(nb.float64)])
def nb_erf(x):
    """
    Numba version of error function. 
    Vectorization is necessary here for the math.erf
    This runs faster than numpy vectorized and the 
    out-of-the-box math.erf
    """
    return math.erf(x)



@nb.vectorize([nb.float32(nb.float32),
nb.float64(nb.float64)])
def nb_erfc(x):
    """
    Numba version of complementary error function
    Vectorization is necessary here for the math.erfc
    This runs faster than numpy vectorized and the 
    out-of-the-box math.erfc
    """
    return math.erfc(x)
