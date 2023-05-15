import re
import sys

from .base import WaveformCodec
from .radware import RadwareSigcompress  # noqa: F401
from .varlen import ULEB128ZigZagDiff  # noqa: F401


def str2wfcodec(expr: str) -> WaveformCodec:
    """Eval strings containing :class:`.WaveformCodec` declarations.

    Simple tool to avoid using :func:`eval`. Used to read
    :class:`.WaveformCodec` declarations configured in JSON files.
    """
    match = re.match(r"(\w+)\((.*)\)", expr.strip())
    if match is None:
        raise ValueError(f"invalid WaveformCodec expression '{expr}'")

    match = match.groups()
    codec = getattr(sys.modules[__name__], match[0].strip())
    args = {}

    if match[1]:
        for items in match[1].split(","):
            sp = items.split("=")
            if len(sp) != 2:
                raise ValueError(f"invalid WaveformCodec expression '{expr}'")

            try:
                args[sp[0].strip()] = float(sp[1].strip())
            except ValueError:
                args[sp[0].strip()] = sp[1].strip().strip("'\"")

    return codec(**args)
