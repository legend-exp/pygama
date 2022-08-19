class NumbaDefaults:
    """Bare-bones class to store some Numba default options."""

    def __init__(self) -> None:
        self.cache: bool = True
        self.boundscheck: bool = False


numba_defaults = NumbaDefaults()
numba_defaults_kwargs = numba_defaults.__dict__
