class NumbaDefaults:
    """Bare-bones class to store some Numba default options.

    Examples
    --------
    Set all default option values at once by expanding the provided dictionary:

    >>> from numba import guvectorize
    >>> from pygama.dsp.utils import numba_defaults_kwargs as nb_kwargs
    >>> @guvectorize([], "", **nb_kwargs, nopython=True) # def proc(...): ...

    Customize one argument but still set defaults for the others:

    >>> from pygama.dsp.utils import numba_defaults as nb_defaults
    >>> @guvectorize([], "", cache=nb_defaults.cache, boundscheck=True) # def proc(...): ...
    """

    def __init__(self) -> None:
        self.cache: bool = True
        self.boundscheck: bool = False


numba_defaults = NumbaDefaults()
numba_defaults_kwargs = numba_defaults.__dict__
