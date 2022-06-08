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

from ._distributions.gauss import gauss_norm, gauss_cdf, gauss_pdf 
from ._distributions.step import unnorm_step_pdf, step_pdf, step_cdf 
from ._distributions.gauss_step import gauss_step_pdf, gauss_step_cdf, extended_gauss_step_pdf
from ._distributions.gauss_on_background import gauss_uniform, gauss_linear 
from ._distributions.gauss_with_tail import gauss_tail_pdf, gauss_tail_cdf, gauss_with_tail_pdf, gauss_with_tail_cdf, gauss_tail_approx, gauss_tail_integral, gauss_tail_norm 
from ._distributions.radford import radford_pdf, radford_cdf, extended_radford_pdf 
from ._distributions.double_gauss import Am_double, extended_Am_double, double_gauss_pdf, extended_double_gauss_pdf  