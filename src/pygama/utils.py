import os
from collections.abc import MutableMapping
from typing import Any, Iterator


def getenv_bool(name: str, default: bool = False) -> bool:
    """Get environment value as a boolean, returning True for 1, t and true
    (caps-insensitive), and False for any other value and default if undefined.
    """
    val = os.getenv(name)
    if not val:
        return default
    elif val.lower() in ("1", "t", "true"):
        return True
    else:
        return False


class NumbaPygamaDefaults(MutableMapping):
    """Bare-bones class to store some Numba default options. Defaults values
    are set from environment variables. Useful for the pygama.math distributions

    Examples
    --------
    Set all default option values for a numba wrapped function at once by expanding the
    provided dictionary:

    >>> from numba import njit
    >>> from pygama.utils import numba_math_defaults_kwargs as nb_kwargs
    >>> @njit([], "", **nb_kwargs, nopython=True) # def dist(...): ...

    Customize one argument but still set defaults for the others:

    >>> from pygama.utils import numba_math_defaults as nb_defaults
    >>> @njit([], "", **nb_defaults(cache=False) # def dist(...): ...

    Override global options at runtime:

    >>> from pygama.utils import numba_math_defaults
    >>> # must set options before explicitly importing pygama.math.distributions!
    >>> numba_math_defaults.cache = False
    """

    def __init__(self) -> None:
        self.parallel: bool = getenv_bool("PYGAMA_PARALLEL", default=True)
        self.fastmath: bool = getenv_bool("PYGAMA_FAST", default=True)

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


numba_math_defaults = NumbaPygamaDefaults()
numba_math_defaults_kwargs = numba_math_defaults
