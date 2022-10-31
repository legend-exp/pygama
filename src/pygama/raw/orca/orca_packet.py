"""
Provides convenience functions for working with ORCA packets.

An ORCA packet is represented by a one-dimensional :class:`numpy.ndarray` of
type :class:`numpy.uint32`.
"""
from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt

log = logging.getLogger(__name__)
OrcaPacket = npt.NDArray[np.uint32]


def is_short(packet: OrcaPacket) -> bool:
    return bool(packet[0] >> 31)


def get_n_words(packet: OrcaPacket) -> int:
    if is_short(packet):
        return 1
    return packet[0] & 0x3FFFF


def get_data_id(packet: OrcaPacket, shift: bool = True) -> np.uint32:
    # in the header the data_id appears unshifted, but in pygama we use it
    # shifted by default so that it has sensible values when printed / checked
    if is_short(packet):
        if shift:
            return (packet[0] & 0xFC000000) >> 26
        else:
            return packet[0] & 0xFC000000
    if shift:
        return (packet[0] & 0xFFFC0000) >> 18
    return packet[0] & 0xFFFC0000


def hex_dump(
    packet: OrcaPacket,
    shift_data_id: bool = True,
    print_n_words: bool = False,
    max_words: int = np.inf,
    as_int: bool = False,
    as_short: bool = False,
    id_dict: dict = None,
    use_logging: bool = True,
) -> None:

    dump_cmd = print  # noqa: T202
    if use_logging:
        dump_cmd = log.debug

    data_id = get_data_id(packet, shift=shift_data_id)
    n_words = get_n_words(packet)
    if id_dict is not None:
        if data_id in id_dict:
            heading = f"{id_dict[data_id]} (data ID = {data_id})"
        else:
            heading = f"[unknown] (data ID = {data_id})"
    else:
        heading = f"data ID = {data_id}"
    if print_n_words:
        dump_cmd(f"{heading}: {n_words} words")
    else:
        dump_cmd(f"{heading}:")
        n_to_print = int(np.minimum(n_words, max_words))
        pad = int(np.ceil(np.log10(n_to_print)))
        for i in range(n_to_print):
            line = f"{str(i).zfill(pad)}"
            line += " {0:#0{1}x}".format(packet[i], 10)
            if data_id == 0 and i > 1:
                line += f" {packet[i:i+1].tobytes().decode()}"
            if as_int:
                line += f" {packet[i]}"
            if as_short:
                line += f" {np.frombuffer(packet[i:i+1].tobytes(), dtype='uint16')}"
            dump_cmd(line)
