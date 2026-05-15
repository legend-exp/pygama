"""
Utilities for working with bit-aggregation columns produced by
:func:`.build_hit`.
"""

from __future__ import annotations

from collections.abc import Iterable

import awkward as ak
import lgdo
import numpy as np
from numpy.typing import ArrayLike


def unpack_bitmask(
    aggregation: lgdo.LGDO | ArrayLike,
    bit_names: Iterable[str] | None = None,
) -> ak.Array:
    """Unpack a bit-aggregation column into an awkward record array.

    The returned array has one boolean field per bit, with field names taken
    from `bit_names` or from the ``bit_names`` attribute of `aggregation` when
    it is an :class:`lgdo.LGDO`. Bit ``i`` maps to ``bit_names[i]``, matching
    the encoding used by :func:`.build_hit`.

    Works on flat columns (e.g. :class:`lgdo.Array`) as well as jagged ones
    (e.g. :class:`lgdo.VectorOfVectors`): the returned record array preserves
    the input shape.

    Parameters
    ----------
    aggregation
        a bit-aggregation column. Either an :class:`lgdo.LGDO` carrying a
        ``bit_names`` attribute, or a plain integer numpy/awkward array — in
        the latter case `bit_names` must be passed explicitly.
    bit_names
        ordered names for each bit. If ``None`` and `aggregation` is an
        :class:`lgdo.LGDO`, names are read from its ``bit_names`` attribute
        (a comma-separated string).
    """
    if isinstance(aggregation, lgdo.LGDO):
        values = aggregation.view_as("ak")
        if bit_names is None:
            attr = aggregation.attrs.get("bit_names")
            if attr is None:
                msg = (
                    "aggregation has no 'bit_names' attribute; pass bit_names "
                    "explicitly"
                )
                raise ValueError(msg)
            bit_names = attr.split(",")
    else:
        if bit_names is None:
            msg = "bit_names must be provided when aggregation is not an LGDO"
            raise ValueError(msg)
        values = (
            aggregation
            if isinstance(aggregation, ak.Array)
            else np.asarray(aggregation)
        )

    names = list(bit_names)
    fields = {name: ((values >> i) & 1) == 1 for i, name in enumerate(names)}
    return ak.zip(fields)
