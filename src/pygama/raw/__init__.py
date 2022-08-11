"""
The primary function for data conversion into raw-tier LH5 files is
:func:`.build_raw`. This is a one-to many function: one input DAQ file can
generate one or more output raw files. Control of which data ends up in which
files, and in which HDF5 groups inside of each file, is controlled via
:mod:`.raw_buffer` (see below). If no raw buffers specification is specified,
all decoded data should be written to a single output file, with all fields
from each hardware decoder in their own output table.

Currently we support the following hardware:

* FlashCam ADC (requires `fcutils <https://github.com/legend-exp/pyfcutils>`_)
* FlashCam ADC read out with `ORCA <https://github.com/unc-enap/Orca>`_
"""

from pygama.raw.build_raw import build_raw

__all__ = ["build_raw"]
