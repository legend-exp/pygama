from __future__ import annotations

from pygama.raw.data_decoder import DataDecoder
from pygama.raw.orca.orca_header import OrcaHeader


class OrcaDecoder(DataDecoder):
    """Base class for ORCA decoders.

    Mostly here to provide a standard interface for setting the header during
    initialization.
    """

    def __init__(self, header: OrcaHeader = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.header = None
        if header:
            self.set_header(header)  # allow for derived class function to be called

    def set_header(self, header: OrcaHeader) -> None:
        """Setter for headers. Overload to set card parameters, etc."""
        self.header = header


def get_ccc(crate: int, card: int, channel: int) -> int:
    """Define a standard hash for (crate, card, channel) to integer."""
    return (crate << 9) + ((card & 0x1F) << 4) + (channel & 0xF)


def get_crate(ccc: int) -> int:
    """Get crate number back from hash value.

    See Also
    --------
    .get_ccc
    """
    return ccc >> 9


def get_card(ccc: int) -> int:
    """Get card number back from hash value.

    See Also
    --------
    .get_ccc
    """
    return (ccc >> 4) & 0x1F


def get_channel(ccc: int) -> int:
    """Get channel number back from hash value.

    See Also
    --------
    .get_ccc
    """
    return ccc & 0xF
