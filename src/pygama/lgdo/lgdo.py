from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LGDO(ABC):
    """Abstract base class representing a LEGEND Data Object (LGDO)."""

    @abstractmethod
    def __init__(self, attrs: dict[str, Any] | None = None) -> None:
        self.attrs = {} if attrs is None else dict(attrs)

        if "datatype" in self.attrs:
            if self.attrs["datatype"] != self.form_datatype():
                raise ValueError(
                    f"datatype attribute ({self.attrs['datatype']}) does "
                    f"not match class datatype ({self.form_datatype()})!"
                )
        else:
            self.attrs["datatype"] = self.form_datatype()

    @abstractmethod
    def datatype_name(self) -> str:
        """The name for this LGDO's datatype attribute."""
        pass

    @abstractmethod
    def form_datatype(self) -> str:
        """Return this LGDO's datatype attribute string."""
        pass

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"(attrs={repr(self.attrs)})"
