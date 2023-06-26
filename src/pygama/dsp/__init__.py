from warnings import warn

from dspeed import *  # noqa: F403, F401

warn(
    "pygama.dsp has moved to its own package, dspeed. "
    "Please replace 'import pygama.dsp' with 'import dspeed'. "
    "pygama.dsp will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)
