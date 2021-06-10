"""
Contains a list of dsp processors used by the legend experiment, implemented
using numba's guvectorize to implement numpy's ufunc interface. In other words,
all of the functions are void functions whose outputs are given as parameters.
The ufunc interface provides additional information about the function
signatures that enables broadcasting the arrays and SIMD processing. Thanks to
the ufunc interface, they can also be called to return a numpy array, but if
 this is done, memory will be allocated for this array, slowing things down.
"""

# I think there's a way to do this recursively, but I'll figure it out later...
from ._processors.mean_stdev import mean_stdev
from ._processors.pole_zero import pole_zero, double_pole_zero
from ._processors.trap_filter import trap_filter
from ._processors.current import avg_current
from ._processors.asym_trap_filter import asymTrapFilter
from ._processors.fixed_time_pickoff import fixed_time_pickoff
from ._processors.trap_norm import trap_norm
from ._processors.trap_pickoff import trap_pickoff
from ._processors.time_point_frac import time_point_frac
from ._processors.time_point_thresh import time_point_thresh
from ._processors.linear_fit import linear_fit
from ._processors.zac_filter import zac_filter
from ._processors.param_lookup import param_lookup
from ._processors.cusp_filter import cusp_filter
from ._processors.fftw import dft, inv_dft, psd
from ._processors.linear_slope_fit import linear_slope_fit
from ._processors.log_check import log_check
from ._processors.min_max import min_max
from ._processors.presum import presum