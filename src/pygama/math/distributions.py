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

from pygama.math.functions.crystall_ball import nb_xtalball_cdf, nb_xtalball_pdf
from pygama.math.functions.gauss import (
    nb_gauss,
    nb_gauss_cdf,
    nb_gauss_norm,
    nb_gauss_pdf,
)
from pygama.math.functions.gauss_on_background import nb_gauss_linear, nb_gauss_uniform
from pygama.math.functions.gauss_step import (
    nb_gauss_step,
    nb_gauss_step_cdf,
    nb_gauss_step_pdf,
)
from pygama.math.functions.gauss_with_tail import (
    nb_exgauss,
    nb_gauss_tail_approx,
    nb_gauss_tail_cdf,
    nb_gauss_tail_integral,
    nb_gauss_tail_norm,
    nb_gauss_with_tail_cdf,
    nb_gauss_with_tail_pdf,
)
from pygama.math.functions.hpge_peak import (
    nb_extended_hpge_peak_pdf,
    nb_hpge_peak_cdf,
    nb_hpge_peak_pdf,
)
from pygama.math.functions.polynomial import nb_poly
from pygama.math.functions.step import nb_step_cdf, nb_step_pdf, nb_unnorm_step_pdf
from pygama.math.functions.triple_gauss import (
    nb_triple_gauss_double_step,
    nb_triple_gauss_double_step_pdf,
)
