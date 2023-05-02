from __future__ import annotations

import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class WaveformCodec:
    """Base class identifying a waveform compression algorithm.

    The `self.codec` property returns a string identifier suitable for labeling
    encoded data on disk. This identifier is constant for all class instances.

    Note
    ----
    This is an abstract type. The user must provided a concrete subclass.
    """

    @property
    def codec(self):
        """The waveform codec string identifier.

        Will be attached as an attribute to the encoded Waveform values.
        """
        return re.sub("(?<!^)(?=[A-Z])", "_", type(self).__name__).lower()

    def asdict(self):
        """Return the dataclass fields as dictionary."""
        return {"codec": self.codec} | asdict(self)
