"""
The pygama signal processing framework is contained in the :mod:`pygama.dsp`
submodule. This code is responsible for running a variety of discrete signal
processors on data, producing tier 2 (dsp) files from tier 1 (raw) files. The
main contents of this submodule are:

* processors: A collection of numba functions that perform individual DSP
  transforms and reductions on our data. Individual processors are held in the
  `_processors` directory, which is where contributors will write new
  processors for inclusion in the analysis.
* :class:`ProcessingChain`: A class that manages and efficiently
  runs a list of DSP processors
"""
