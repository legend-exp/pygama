"""
Pygama works with "LEGEND Data Objects" (LGDO) defined in the `LEGEND data
format specification <https://github.com/legend-exp/legend-data-format-specs>`_.
This submodule serves as the Python implementation of that specification. The
general strategy for the implementation is to dress standard Python and NumPy
objects with an ``attr`` dictionary holding LGDO metadata, plus some convenience
functions. The basic data object classes are:

* :class:`.Scalar`: typed Python scalar. Access data via the :attr:`value`
  attribute
* :class:`.Array`: basic NumPy :class:`ndarray`. Access data via the
  :attr:`nda` attribute.
* :class:`.FixedSizeArray`: basic NumPy :class:`ndarray`. Access data via the
  :attr:`nda` attribute.
* :class:`.ArrayOfEqualSizedArrays`: multi-dimensional NumPy :class:`ndarray`.
  Access data via the :attr:`nda` attribute.
* :class:`.VectorOfVectors`: a variable length array of variable length arrays.
  Implemented as a pair of :class:`.Array`: :attr:`flattened_data` holding the
  raw data, and :attr:`cumulative_length` whose ith element is the sum of the
  lengths of the vectors with ``index <= i``
* :class:`.Struct`: a dictionary containing LGDO objects. Derives from
  :class:`dict`
* :class:`.Table`: a :class:`.Struct` whose elements ("columns") are all array
  types with the same length (number of rows)

Currently the primary on-disk format for LGDO object is LEGEND HDF5 (LH5) files. IO
is done via the class :class:`.lh5_store.LH5Store`. LH5 files can also be
browsed easily in python like any `HDF5 <https://www.hdfgroup.org>`_ file using
`h5py <https://www.h5py.org>`_.
"""

from .array import Array
from .arrayofequalsizedarrays import ArrayOfEqualSizedArrays
from .fixedsizearray import FixedSizeArray
from .lgdo_utils import *
from .lh5_store import LH5Store, load_dfs, load_nda, ls
from .scalar import Scalar
from .struct import Struct
from .table import Table
from .vectorofvectors import VectorOfVectors
from .waveform_table import WaveformTable
