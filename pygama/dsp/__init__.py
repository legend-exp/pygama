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

Generally we have *calculators*, which take a block and return a column
(single-valued), and *transforms*, which take a block and return another block.

DSP is performed by extracting a table of raw data including waveforms and
passing it to the :class:`pygama.dsp.processing_chain.ProcessingChain`. The primary
function for DSP is :func:`pygama.dsp.build_dsp.raw_to_dsp`.

The DSP and other routines can make use of an analysis parameters database,
which is a `JSON <https://www.json.org>`_-formatted file read in as a python
dictionary. It can be sent to the DSP routines to load optimal parameters for a
given channel.
"""
