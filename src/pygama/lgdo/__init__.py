"""
Pygama works with "LEGEND Data Objects" (LGDO) defined in the `LEGEND data
format specification <https://github.com/legend-exp/legend-data-format-specs>`_.
This subpackage serves as the Python implementation of that specification. The
general strategy for the implementation is to dress standard Python and NumPy
objects with an ``attr`` dictionary holding LGDO metadata, plus some convenience
functions. The basic data object classes are:

* :class:`.LGDO`: abstract base class for all LGDOs
* :class:`.Scalar`: typed Python scalar. Access data via the :attr:`value`
  attribute
* :class:`.Array`: basic :class:`numpy.ndarray`. Access data via the
  :attr:`nda` attribute.
* :class:`.FixedSizeArray`: basic :class:`numpy.ndarray`. Access data via the
  :attr:`nda` attribute.
* :class:`.ArrayOfEqualSizedArrays`: multi-dimensional :class:`numpy.ndarray`.
  Access data via the :attr:`nda` attribute.
* :class:`.VectorOfVectors`: a variable length array of variable length arrays.
  Implemented as a pair of :class:`.Array`: :attr:`flattened_data` holding the
  raw data, and :attr:`cumulative_length` whose ith element is the sum of the
  lengths of the vectors with ``index <= i``
* :class:`.VectorOfEncodedVectors`: an array of variable length *encoded*
  arrays. Implemented as a :class:`.VectorOfVectors` :attr:`encoded_data`
  holding the encoded vectors and an :class:`.Array` :attr:`decoded_size`
  specifying the size of each decoded vector. Mainly used to represent a list
  of compressed waveforms.
* :class:`.ArrayOfEncodedEqualSizedArrays`: an array of equal sized encoded
  arrays. Similar to :class:`.VectorOfEncodedVectors` except for
  :attr:`decoded_size`, which is now a scalar.
* :class:`.Struct`: a dictionary containing LGDO objects. Derives from
  :class:`dict`
* :class:`.Table`: a :class:`.Struct` whose elements ("columns") are all array
  types with the same length (number of rows)

Currently the primary on-disk format for LGDO object is LEGEND HDF5 (LH5) files. IO
is done via the class :class:`.lh5_store.LH5Store`. LH5 files can also be
browsed easily in python like any `HDF5 <https://www.hdfgroup.org>`_ file using
`h5py <https://www.h5py.org>`_.
"""

from pygama.lgdo.waveform_table import WaveformTable

from .array import Array
from .arrayofequalsizedarrays import ArrayOfEqualSizedArrays
from .encoded import ArrayOfEncodedEqualSizedArrays, VectorOfEncodedVectors
from .fixedsizearray import FixedSizeArray
from .lgdo import LGDO
from .lgdo_utils import copy
from .lh5_store import LH5Iterator, LH5Store, load_dfs, load_nda, ls, show
from .scalar import Scalar
from .struct import Struct
from .table import Table
from .vectorofvectors import (
    VectorOfVectors,
    build_cl,
    explode,
    explode_arrays,
    explode_cl,
)

__all__ = [
    "Array",
    "ArrayOfEqualSizedArrays",
    "ArrayOfEncodedEqualSizedArrays",
    "FixedSizeArray",
    "LGDO",
    "Scalar",
    "Struct",
    "Table",
    "VectorOfVectors",
    "VectorOfEncodedVectors",
    "WaveformTable",
    "LH5Iterator",
    "LH5Store",
    "load_dfs",
    "load_nda",
    "ls",
    "show",
    "build_cl",
    "explode",
    "explode_arrays",
    "explode_cl",
    "copy",
]

import numpy as np

np.set_printoptions(threshold=10)
