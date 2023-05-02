from __future__ import annotations

import logging

from .. import lgdo
from . import radware, varlen
from .base import WaveformCodec

log = logging.getLogger(__name__)


def encode(
    obj: lgdo.VectorOfVectors | lgdo.ArrayOfEqualsizedArrays,
    codec: WaveformCodec | str = None,
) -> lgdo.VectorOfEncodedVectors | lgdo.ArrayOfEncodedEqualSizedArrays:
    """Encode LGDOs with `codec`.

    Defines behaviors for each implemented waveform encoding algorithm.

    Parameters
    ----------
    obj
        LGDO array type.
    codec
        algorithm to be used for encoding.
    """
    log.debug(f"encoding {repr(obj)} with {codec}")

    if _is_codec(codec, radware.RadwareSigcompress):
        enc_obj = radware.encode(obj, shift=codec.codec_shift)
    elif _is_codec(codec, varlen.ULEB128ZigZagDiff):
        enc_obj = varlen.encode(obj)
    else:
        raise ValueError(f"'{codec}' not supported")

    enc_obj.attrs |= codec.asdict()

    return enc_obj


def decode(
    obj: lgdo.VectorOfEncodedVectors | lgdo.ArrayOfEncodedEqualSizedArrays,
) -> lgdo.VectorOfVectors | lgdo.ArrayOfEqualsizedArrays:
    """Decode encoded LGDOs.

    Defines decoding behaviors for each implemented waveform encoding
    algorithm. Expects to find the codec (and its parameters) the arrays where
    encoded with among the LGDO attributes.

    Parameters
    ----------
    obj
        LGDO array type.
    """
    if "codec" not in obj.attrs:
        raise RuntimeError(
            "object does not carry any 'codec' attribute, I don't know how to decode it"
        )

    codec = obj.attrs["codec"]
    log.debug(f"decoding {repr(obj)} with {codec}")

    if _is_codec(codec, radware.RadwareSigcompress):
        return radware.decode(obj, shift=int(obj.attrs.get("codec_shift", 0)))
    elif _is_codec(codec, varlen.ULEB128ZigZagDiff):
        return varlen.decode(obj)
    else:
        raise ValueError(f"'{codec}' not supported")


def _is_codec(ident: WaveformCodec | str, codec) -> bool:
    if isinstance(ident, WaveformCodec):
        return isinstance(ident, codec)
    elif isinstance(ident, str):
        return ident == codec().codec
    else:
        raise ValueError("input must be WaveformCodec object or string identifier")
