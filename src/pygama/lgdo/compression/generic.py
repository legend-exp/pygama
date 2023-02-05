from __future__ import annotations

from ..lgdo import LGDO
from . import radware
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
        enc_obj.attrs |= codec.asdict()
        return enc_obj
    else:
        raise ValueError(f"'{codec}' not supported")


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
        if "codec_shift" in obj.attrs:
            return radware.decode(obj, shift=int(obj.attrs["codec_shift"]))
        else:
            raise RuntimeError(
                "object does not carry any 'codec_shift' attribute, needed to radware_decode"
            )
    else:
        raise ValueError(f"'{codec}' not supported")
