"""Routines from reading and writing LEGEND Data Objects in HDF5 files.

Currently the primary on-disk format for LGDO object is LEGEND HDF5 (LH5) files. IO
is done via the class :class:`.store.LH5Store`. LH5 files can also be
browsed easily in python like any `HDF5 <https://www.hdfgroup.org>`_ file using
`h5py <https://www.h5py.org>`_.
"""

from .iterator import LH5Iterator
from .store import LH5Store
from .utils import load_dfs, load_nda, ls, show

__all__ = [
    "LH5Iterator",
    "LH5Store",
    "load_dfs",
    "load_nda",
    "ls",
    "show",
]
