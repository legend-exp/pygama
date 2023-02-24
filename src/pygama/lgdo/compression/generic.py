from __future__ import annotations

from ..lgdo import LGDO
from . import google, radware
from .base import WaveformCodec


def encode_array(obj: LGDO, codec: WaveformCodec) -> LGDO:
    """Encode arrays with `codec`.

    Defines behaviors for each implemented waveform encoding algorithm.

    Parameters
    ----------
    obj
        LGDO array type.
    codec
        algorithm to be used for encoding.
    """
    if isinstance(codec, radware.RadwareSigcompress):
        enc_obj = radware.encode(obj, shift=codec.codec_shift)
    elif isinstance(codec, google.GoogleProtobuf):
        enc_obj = google.encode(obj, zigzag=codec.zigzag)
    else:
        raise ValueError(f"'{codec}' not supported")

    enc_obj.attrs |= codec.asdict()
    return enc_obj


def decode_array(obj: LGDO) -> LGDO:
    """Decode encoded arrays.

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

    if codec == "radware_sigcompress":
        return radware.decode(obj, shift=int(obj.attrs.get("codec_shift", 0)))
    elif codec == "google_protobuf":
        return google.decode(obj, zigzag=obj.attrs.get("zigzag", False))
    else:
        raise ValueError(f"'{codec}' not supported")
