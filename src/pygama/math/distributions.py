r"""
Contains a list of distribution functions, some implemented using Numba's
:func:`numba.guvectorize` to implement NumPy's :class:`numpy.ufunc` interface,
and others are implemented without Numba. In other words, all of the Numba wrapped
functions are void functions whose outputs are given as parameters.
The :class:`~numpy.ufunc` interface provides additional
information about the function signatures that enables broadcasting the arrays
and SIMD processing. Thanks to the :class:`~numpy.ufunc` interface, they can
also be called to return a NumPy array, but if this is done, memory will be
allocated for this array, slowing things down.
"""

from pygama.math.functions.triple_gauss import (
    triple_gauss_double_step_pdf,
    triple_gauss_double_step
)
from pygama.math.functions.gauss import gauss_cdf, gauss_norm, gauss_pdf, gauss
from pygama.math.functions.gauss_on_background import gauss_linear, gauss_uniform
from pygama.math.functions.gauss_step import (
    gauss_step,
    gauss_step_cdf,
    gauss_step_pdf,
)
from pygama.math.functions.gauss_with_tail import (
    gauss_tail_approx,
    gauss_tail_cdf,
    gauss_tail_integral,
    gauss_tail_norm,
    exgauss,
    gauss_with_tail_cdf,
    gauss_with_tail_pdf,
)
from pygama.math.functions.hpge_peak import extended_hpge_peak_pdf, hpge_peak_cdf, hpge_peak_pdf
from pygama.math.functions.step import step_cdf, step_pdf, unnorm_step_pdf
from pygama.math.functions.crystall_ball import xtalball
from pygama.math.functions.polynomial import poly
