import os
from collections.abc import MutableMapping
from typing import Any, Iterator

class NumbaDefaults(MutableMapping):
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
    >>> @guvectorize([], "", **nb_defaults(cache=False) # def proc(...): ...

    Set global options at runtime:

    >>> from pygama.dsp.utils import numba_defaults
    >>> from pygama.dsp import build_dsp
    >>> # must set options before explicitly importing pygama.dsp.processors!
    >>> numba_defaults.cache = False
    >>> numba_defaults.boundscheck = True
    >>> build_dsp(...) # if not explicit, processors imports happen here
    """

    def __init__(self) -> None:
        cache = os.getenv("PYGAMA_CACHE")
        if not cache or not cache.lower() in ('1', 't', 'true'):
            self.cache: bool = False
        else:
            self.cache: bool = True
        self.boundscheck: bool = False

    def __getitem__(self, item: str) -> Any:
        return self.__dict__[item]

    def __setitem__(self, item: str, val: Any) -> None:
        self.__dict__[item] = val

    def __delitem__(self, item: str) -> None:
        del self.__dict__[item]

    def __iter__(self) -> Iterator:
        return self.__dict__.__iter__()

    def __len__(self) -> int:
        return len(self.__dict__)
    
    def __call__(self, **kwargs) -> dict:
        mapping = self.__dict__.copy()
        mapping.update(**kwargs)
        return mapping

    def __str__(self) -> str:
        return str(self.__dict__)

    def __repr__(self) -> str:
        return str(self.__dict__)

numba_defaults = NumbaDefaults()
numba_defaults_kwargs = numba_defaults
