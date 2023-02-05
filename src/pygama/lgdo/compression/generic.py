from __future__ import annotations

from ..lgdo import LGDO
from . import radware
from .base import WaveformCodec


def encode_array(obj: LGDO, encoder: WaveformCodec) -> LGDO:
    if isinstance(encoder, radware.RadwareSigcompress):
        enc_obj = radware.encode(obj, shift=encoder.codec_shift)
        enc_obj.attrs |= encoder.asdict()
        return enc_obj
    else:
        raise ValueError(f"'{encoder}' not supported")


def decode_array(obj: LGDO) -> LGDO:
    if "codec" not in obj.attrs:
        raise RuntimeError(
            "object does not carry any 'codec' attribute, I don't know how to decode it"
        )

    encoder = obj.attrs["codec"]

    if encoder == "radware_sigcompress":
        if "codec_shift" in obj.attrs:
            return radware.decode(obj, shift=int(obj.attrs["codec_shift"]))
        else:
            raise RuntimeError(
                "object does not carry any 'codec_shift' attribute, needed to radware_decode"
            )
    else:
        raise ValueError(f"'{encoder}' not supported")
