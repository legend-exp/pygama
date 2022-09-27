Digital Signal Processing
=========================

Global Numba options
--------------------

Pygama offers the possibility to change the value of some default Numba options
either using environment variables or at runtime. One typical use case is to enable `caching the result of compilation
<https://numba.readthedocs.io/en/stable/user/jit.html?#cache>`_,
which significantly reduces loading times. Numba options globally set by pygama
are defined as attributes of the :class:`~.dsp.utils.NumbaDefaults` class. Have
a look to the documentation for :func:`numba.jit` and :func:`numba.guvectorize`
to learn about their meaning.

.. note::
   Some special processors override default option values.

The environment variables that can be set are:

:``PYGAMA_CACHE``: Set caching behavior (default false)
:``PYGAMA_BOUNDSCHECK``: Set automatic bounds checking (default false)

Here's an example of how global option customization can achieved in user
scripts:

.. code-block:: python

    from pygama.dsp.utils import numba_defaults
    from pygama.dsp import build_dsp

    # must set options before explicitly importing pygama.dsp.processors!
    numba_defaults.cache = True

    # processors imports happen here, if not explicitly done before
    build_dsp(...)

Command line interface
----------------------

A command line interface to :func:`~.dsp.build_dsp.build_dsp` is available
through the ``pygama`` executable via the ``build-dsp`` sub-command. This can
be used to quickly run signal processing routines without custom scripting.
Here are some examples of what can be achieved:

.. code-block:: console

    $ pygama build-dsp --help  # display usage and exit

Convert files and save them in the original directory with the same filenames
(but new extension ``_dsp.lh5``):

.. code-block:: console

    $ pygama [-v] build-dsp --config dsp-config.json raw/*.lh5  # increase verbosity with -v
    $ pygama build-dsp --overwrite -c dsp-config.json raw/*.lh5  # overwrite output files
    $ # set maximum number of rows to be considered from each file
    $ pygama build-dsp --max-rows 100 -c dsp-config.json raw/*.lh5

The signal processors are configured with the ``dsp-config.json`` JSON file
(refer to the :func:`~.dsp.build_dsp.build_dsp` documentation for details).

.. seealso::
   See :func:`~.dsp.build_dsp.build_dsp` and ``pygama build-dsp --help`` for a
   full list of conversion options.

Writing custom processors
-------------------------

.. code-block:: python

    # 1) Import Python modules
    #
    from __future__ import annotations

    import numpy as np
    from numba import guvectorize

    from pygama.dsp.errors import DSPFatal
    from pygama.dsp.utils import numba_defaults_kwargs as nb_kwargs

    # 2) Provide instructions to Numba
    #
    # Documentation about Numba guvectorize decorator:
    # https://numba.pydata.org/numba-doc/latest/user/vectorize.html#the-guvectorize-decorator
    #
    # Notes:
    # - Set default Numba arguments by expanding dsp.utils.numba_defaults_kwargs (see below)
    # - If you need to customize the value of one default argument do it, and
    #   use numba_defaults from dsp.utils to set up the remaining arguments:
    #       @guvectorize(..., cache=numba_defaults.cache, boundscheck=True)
    # - Use two declarations, one for 32-bit variables and one for 64-bit variables
    # - Do not use "int" as it does not support an NaN value
    # - Use [:] for all output parameters
    #
    @guvectorize(["void(float32[:], float32, float32, float32[:])",
                  "void(float64[:], float64, float64, float64[:])"],
                  "(n),(),()->()", **nb_kwargs)

    # 3) Define the processor interface
    #
    # Notes:
    # - Add the "_in"/"_out" suffix to the name of the input/output variables
    # - Use "w_" for waveforms, "t_" for indexes, "a_" for amplitudes
    # - Use underscore casing for the name of the processor and variables, e.g.,
    #   "a_trap_energy_in" or "t_trigger_in"
    # - use type annotations
    #
    def the_processor_template(w_in: np.ndarray,
                               t_in: float,
                               a_in: float,
                               w_out: np.ndarray,
                               t_out: float) -> None:

        # 4) Document the algorithm
        #
        """One-liner description of the processor.

        Add here a more detailed description of what the processor does.
        Document input parameters in the "Parameters" section. Add a JSON
        example for ProcessingChain configuration in the last section.

        Parameters
        ----------
        w_in
            the input waveform.
        t_in
            a scalar parameter in the time domain
        a_in
            a scalar parameter in the amplitude domain
        w_out
            the output waveform
        t_out
            an output scalar value in the time domain

        JSON Configuration Example
        --------------------------

        .. code-block :: json

            "wf_bl": {
                "function": "the_processor_template",
                "module": "pygama.dsp.processors",
                "args": ["waveform", "t_a", "a_b", "wf_filtered", "t_result"],
                "unit": "ADC"
            }
        """

        # 5) Initialize output parameters
        #
        # Notes:
        # - All output parameters should be initializes to a NaN.  If a processor
        #   fails, its output parameters should have the default NaN value
        # - Use np.nan for both variables and arrays
        #
        w_out[:] = np.nan # use [:] for arrays
        t_out[0] = np.nan # use [0] for scalar parameters

        # 6) Check inputs
        #
        # There are two kinds of checks:
        # - NaN checks.  A processor might depend on others, i.e., its input
        #   parameters are the output parameters of an other processors.  When a
        #   processor fails, all processors depending on its output should not run.
        #   Thus, skip this processor if a NaN value is detected and return NaN
        #   output parameters to propagate the failure throughout the processing chain.
        # - In-range checks.  Check if indexes are within 0 and len(waveform),
        #   amplitudes are positive, etc.  A failure of this check implies errors in
        #   the DSP JSON config file.  Abort the analysis immediately.
        #
        if np.isnan(w_in).any() or np.isnan(t_in) or np.isnan(a_in):
            return

        if a_in < 0:
            raise DSPFatal('The error message goes here')

        # 7) Algorithm
        #
        # Loop over waveforms by using a "for i in range(.., .., ..)" instruction.
        # Avoid loops based on a while condition which might lead to segfault.  Avoid
        # also enumerate/ndenumerate to keep code as similar as possible among all
        # processors.
        #
        # Example of an algorithm to find the first index above "t_in" in which the
        # signal crossed the value "a_in"
        #
        for i in range(t_in, 1, 1):
            if w_in[i] >= a_in and w_in[i-1] < a_in:
                t_out[0] = i
                return
