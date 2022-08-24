class NumbaDefaults:
    """Bare-bones class to store some Numba default options.

    Examples
    --------
    Set all default option values for a processor at once by expanding the
    provided dictionary:

    >>> from numba import guvectorize
    >>> from pygama.dsp.utils import numba_defaults_kwargs as nb_kwargs
    >>> @guvectorize([], "", **nb_kwargs, nopython=True) # def proc(...): ...

    Customize one argument but still set defaults for the others:

    >>> from pygama.dsp.utils import numba_defaults as nb_defaults
    >>> @guvectorize([], "", cache=True, boundscheck=nb_defaults.boundscheck) # def proc(...): ...

    Set global options at runtime:

    >>> from pygama.dsp.utils import numba_defaults
    >>> from pygama.dsp import build_dsp
    >>> # must set options before explicitly importing pygama.dsp.processors!
    >>> numba_defaults.cache = False
    >>> numba_defaults.boundscheck = True
    >>> build_dsp(...) # if not explicit, processors imports happen here
    """

    def __init__(self) -> None:
        self.cache: bool = False
        self.boundscheck: bool = False


numba_defaults = NumbaDefaults()
numba_defaults_kwargs = numba_defaults.__dict__
