"""
The primary function for data conversion into raw-tier LH5 files is
:func:`.build_raw`. This is a one-to many function: one input DAQ file can
generate one or more output raw files. Control of which data ends up in which
files, and in which HDF5 groups inside of each file, is controlled via
:mod:`.raw_buffer` (see below). If no raw buffers specification is specified,
all decoded data should be written to a single output file, with all fields
from each hardware decoder in their own output table.

Currently we support the following DAQ data formats:

* `FlashCam <https://www.mizzi-computer.de/home>`_
* `CoMPASS <https://www.caen.it/products/compass>`_
* `ORCA <https://github.com/unc-enap/Orca>`_, reading out:

  - FlashCam
  - `Struck SIS3302 <https://www.struck.de/sis3302.htm>`_
  - `Struck SIS3316 <https://www.struck.de/sis3316.html>`_
"""

from pygama.raw.build_raw import build_raw

__all__ = ["build_raw"]
