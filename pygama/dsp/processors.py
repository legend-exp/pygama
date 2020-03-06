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
from ._processors.pole_zero import pole_zero
from ._processors.trap_filter import trap_filter
from ._processors.current import avg_current
