r"""
The pygama signal processing framework is contained in the :mod:`.dsp`
submodule. This code is responsible for running a variety of discrete signal
processors on data, producing tier 2 (dsp) files from tier 1 (raw) files. The
main contents of this submodule are:

* :mod:`.processors`: A collection of Numba functions that perform individual
  DSP transforms and reductions on our data. Here is where contributors will
  write new processors for inclusion in the analysis. Available processors
  include all :class:`numpy.ufunc`\ s as well.
* :class:`.ProcessingChain`: A class that manages and efficiently runs a list
  of DSP processors
* :func:`.build_processing_chain`: A function that builds a :class:`.ProcessingChain`
  using LH5-formatted input and output files, and a JSON configuration file
* :func:`.build_dsp`: A function that runs :func:`.build_processing_chain` to build a
  :class:`.ProcessingChain` from a JSON config file and then processes an input
  file and writes into an output file, using the LH5 file format
"""

from pygama.dsp.build_dsp import build_dsp
from pygama.dsp.processing_chain import ProcessingChain, build_processing_chain

__all__ = ["build_dsp", "ProcessingChain", "build_processing_chain"]
