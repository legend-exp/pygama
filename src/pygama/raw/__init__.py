from warnings import warn

from daq2lh5 import *  # noqa: F403, F401

warn(
    "pygama.raw has moved to its own package, legend-daq2lh5. "
    "Please replace 'import pygama.raw' with 'import daq2lh5'. "
    "pygama.raw will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)
