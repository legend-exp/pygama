from warnings import warn

from lgdo import *  # noqa: F403, F401

warn(
    "pygama.lgdo has moved to its own package, legend-pydataobj. "
    "Please replace 'import pygama.lgdo' with 'import lgdo'. "
    "pygama.lgdo will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)
