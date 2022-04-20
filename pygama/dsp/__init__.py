"""
The pygama signal processing framework is contained in the :mod:`pygama.dsp`
submodule. This code is responsible for running a variety of discrete signal
processors on data, producing tier 2 (dsp) files from tier 1 (raw) files. The
main contents of this submodule are:

* :mod:`pygama.dsp.processors`: A collection of Numba functions that perform
  individual DSP transforms and reductions on our data. Individual processors are
  held in the ``_processors`` directory, which is where contributors will write
  new processors for inclusion in the analysis. Available processors include all
  numpy ufuncs as well.
* :class:`pygama.dsp.ProcessingChain`: A class that manages and efficiently
  runs a list of DSP processors
* :func:`build_processing_chain`: A function that builds a ProcessingChain using lh5 formatted input and output files, and a json configuration file
* :func:`build_dsp`: A function that runs build_processing_chain to build a ProcessingChain from a json config file and then processes an input file and writes into an output file, using the lh5 file format
"""
