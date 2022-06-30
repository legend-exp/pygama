r"""
Contains a list of DSP processors, implemented using Numba's
:func:`numba.guvectorize` to implement NumPy's :class:`numpy.ufunc` interface.
In other words, all of the functions are void functions whose outputs are given
as parameters.  The :class:`~numpy.ufunc` interface provides additional
information about the function signatures that enables broadcasting the arrays
and SIMD processing. Thanks to the :class:`~numpy.ufunc` interface, they can
also be called to return a NumPy array, but if this is done, memory will be
allocated for this array, slowing things down.

The pygama processors use the :class:`~numpy.ufunc` framework, which is
designed to encourage highly performant python practices. These functions have
several advantages:

1. They work with :class:`numpy.array`. NumPy arrays are arrays of same-typed
   objects that are stored adjacently in memory (equivalent to a dynamically
   allocated C-array or a C++ vector). Compared to standard Python lists, they
   perform computations much faster. They are ideal for representing waveforms
   and transformed waveforms.

2. They perform vectorized operations. Vectorization causes commands to
   perform the same operation on all components of an array. For example, the
   :class:`~numpy.ufunc` ``np.add(a, b, out=c)`` (equivalently ``c=a+b``) is
   equivalent to: ::

       for i in range(len(c)): c[i] = a[i] + b[i]

   Loops are slow in python since it is an interpreted language; vectorized
   commands remove the loop and only call the Python interpreter once.

   Furthermore, :class:`~numpy.ufunc`\ s are capable of `broadcasting
   <https://docs.scipy.org/doc/numpy/reference/ufuncs.html#broadcasting>`_
   their dimensions. This involves a safety check to ensure the dimensions of
   ``a`` and ``b`` are compatible sizes. It also will automatically replicate a
   ``size-1`` dimension over another one, enabling the addition of a scalar to
   a vector quantity. This is useful, as it allows us to process multiple
   waveforms at once.

   One of the biggest advantages of vectorized :class:`~numpy.ufunc`\ s is that
   many of them will take advantage of SIMD (same input-multiple data)
   vectorization on a vector-CPU. Modern CPUs typically have 256- or 512-bit
   wide processing units, which can accommodate multiple 32- or 64-bit numbers.
   Programming with these, however, is quite difficult and requires specialized
   commands to be called.  Luckily for us, many NumPy :class`~numpy.ufunc`\ s
   will automatically use these for us, speeding up our code!

3. :class:`~numpy.ufunc`\ s are capable of calculating their output in place,
   meaning they can place calculated values in pre-allocated memory rather than
   allocating and returning new values. This is important because memory
   allocation is one of the slowest processes computers can perform, and should
   be avoided. With :class:`~numpy.ufunc`\ s, this can be done using the out
   keyword in arguments (ex ``numpy.add(a, b, out=c)``, or more succinctly,
   ``numpy.add(a, b, c)``).  While this may seem counterintuitive at first, the
   alternative (``c = np.add(a,b)`` or ``c = a+b``) causes an entirely new
   array to be allocated, with c pointing at that. These array allocations can
   add up very quickly: ``e = a*b + c*d``, for example, would allocate 3
   different arrays: one for ``a*b``, one for ``c*d``, and one for the sum of
   those two. As we write :class:`~numpy.ufunc`\ s, it is important that we try
   to use functions that operate in place as much as possible!
"""

from ._processors.bl_subtract import bl_subtract
from ._processors.convolutions import cusp_filter, t0_filter, zac_filter
from ._processors.fftw import dft, inv_dft, psd
from ._processors.fixed_time_pickoff import fixed_time_pickoff
from ._processors.gaussian_filter1d import gaussian_filter1d
from ._processors.get_multi_local_extrema import get_multi_local_extrema
from ._processors.linear_slope_fit import linear_slope_fit
from ._processors.log_check import log_check
from ._processors.min_max import min_max
from ._processors.moving_windows import (
    avg_current,
    moving_window_left,
    moving_window_multi,
    moving_window_right,
)
from ._processors.multi_a_filter import multi_a_filter
from ._processors.multi_t_filter import multi_t_filter, remove_duplicates
from ._processors.optimize import optimize_1pz, optimize_2pz
from ._processors.param_lookup import param_lookup
from ._processors.pole_zero import double_pole_zero, pole_zero
from ._processors.presum import presum
from ._processors.pulse_injector import inject_exp_pulse, inject_sig_pulse
from ._processors.saturation import saturation
from ._processors.soft_pileup_corr import soft_pileup_corr, soft_pileup_corr_bl
from ._processors.time_point_thresh import time_point_thresh
from ._processors.trap_filters import (
    asym_trap_filter,
    trap_filter,
    trap_norm,
    trap_pickoff,
)
from ._processors.upsampler import upsampler
from ._processors.wiener_filter import wiener_filter
from ._processors.windower import windower
