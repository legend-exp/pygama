from __future__ import annotations

import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True, kw_only=True)
class WaveformCodec:
    @property
    def codec(self):
        return re.sub("(?<!^)(?=[A-Z])", "_", type(self).__name__).lower()

    def asdict(self):
        return {"codec": self.codec} | asdict(self)
