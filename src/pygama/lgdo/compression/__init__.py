r"""Data compression utilities.

This subpackage collects all *pygama* custom data compression (encoding) and
decompression (decoding) algorithms.

Available lossless waveform compression algorithms:

* :class:`.RadwareSigcompress`, a Python port of the C algorithm
  `radware-sigcompress` by D. Radford.
* :class:`.ULEB128ZigZagDiff` variable-length base-128 encoding of waveform
  differences.

All waveform compression algorithms inherit from the :class:`.WaveformCodec`
abstract class.

:func:`~.generic.encode` and :func:`~.generic.decode` provide a high-level
interface for encoding/decoding :class:`~.lgdo.LGDO`\ s.

>>> from pygama.lgdo import WaveformTable, compression
>>> wftbl = WaveformTable(...)
>>> enc_wft = compression.encode(wftable, RadwareSigcompress(codec_shift=-23768)
>>> compression.decode(enc_wft) # == wftbl
"""

from .base import WaveformCodec
from .generic import decode, encode
from .radware import RadwareSigcompress
from .varlen import ULEB128ZigZagDiff

__all__ = [
    "WaveformCodec",
    "encode",
    "decode",
    "RadwareSigcompress",
    "ULEB128ZigZagDiff",
]
