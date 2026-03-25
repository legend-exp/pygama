Package overview
================

*pygama* is a Python package developed by the `LEGEND collaboration
<https://legend-exp.org>`_ for processing and analysing data from
high-purity germanium (HPGe) and liquid-argon (LAr) detector systems.  It
sits at the higher levels of the LEGEND data processing chain and operates
on data that has already been decoded and signal-processed by the companion
packages `legend-daq2lh5 <https://legend-daq2lh5.readthedocs.io>`_ and
`dspeed <https://dspeed.readthedocs.io>`_.

All data is stored and exchanged in the `LEGEND HDF5 (LH5)
<https://legend-pydataobj.readthedocs.io>`_ format via the
:mod:`lgdo` library.

.. _overview-data-tiers:

Data tiers
----------

The LEGEND processing chain is organised as a sequence of *data tiers*, each
one adding progressively higher-level information:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Tier
     - Description
   * - ``raw``
     - Raw digitiser output decoded to LH5 by *legend-daq2lh5*.
   * - ``dsp``
     - Digital-signal-processing (DSP) parameters extracted from waveforms by
       *dspeed*: trapezoidal-filter energies, current amplitudes, timestamps,
       etc.
   * - ``hit``
     - Per-hit derived quantities (calibrated energy, quality-cut flags, …)
       produced from the ``dsp`` tier by :mod:`pygama.hit`.
   * - ``tcm``
     - Time Coincidence Map: a lookup table that groups ``hit``-tier rows from
       different channels into physics events, built by :mod:`pygama.evt`.
   * - ``evt``
     - Event-level quantities aggregated across all channels that contribute to
       a single physics event, built by :mod:`pygama.evt`.

.. _overview-modules:

Main modules
------------

:mod:`pygama` exposes four main sub-packages, each covering a distinct stage
of the processing chain.

.. rubric:: :mod:`pygama.hit` — Hit-tier production

Applies user-defined columnar transformations to ``dsp``-tier tables to
produce the ``hit`` tier.  Expressions are given as strings (evaluated via
:meth:`~lgdo.types.table.Table.eval`) and configured through a JSON
dictionary, making the tier highly configurable without writing Python code.
See :doc:`manuals/hit` for a detailed description.

.. rubric:: :mod:`pygama.evt` — Event building

Groups hit-level data from multiple channels and multiple detector types into
physics events using a Time Coincidence Map (TCM).  The main entry points are
:func:`~pygama.evt.build_tcm`, which builds the TCM from coincident hits, and
:func:`~pygama.evt.build_evt`, which evaluates per-event quantities by
aggregating across channels according to a JSON configuration.  Detector-
specific processors for HPGe, SiPM and LAr veto subsystems live in the
:mod:`pygama.evt.modules` sub-package.
See :doc:`manuals/evt` for a detailed description.

.. rubric:: :mod:`pygama.math` — Mathematical utilities

A collection of statistical distributions, histogram helpers, and fitting
routines used throughout the package.  All probability density and cumulative
distribution functions are JIT-compiled with `Numba
<https://numba.readthedocs.io>`_ for speed.  Binned and unbinned maximum-
likelihood fits are implemented on top of `iminuit
<https://iminuit.readthedocs.io>`_.
See :doc:`manuals/math` for a detailed description.

.. rubric:: :mod:`pygama.pargen` — Parameter generation

Routines for calibrating detector parameters from data: HPGe energy
calibration, amplitude-over-energy (A/E) multi-site-event discrimination,
late-charge (LQ) cut calibration, DSP-filter optimisation, and data-quality
cuts.  Results are typically stored as JSON parameter files that are then
consumed by :mod:`pygama.hit`.
See :doc:`manuals/pargen` for a detailed description.

.. _overview-flow:

Data flow summary
-----------------

The diagram below shows how the four main sub-packages interact within the
overall processing chain::

    legend-daq2lh5          dspeed               pygama
    ┌─────────────┐       ┌──────────┐     ┌─────────────────────────────────┐
    │  raw tier   │──────▶│ dsp tier │────▶│  hit tier   (pygama.hit)        │
    │  (decoded)  │       │ (wvf     │     │  (calibrated quantities)        │
    └─────────────┘       │  params) │     └────────────────┬────────────────┘
                          └──────────┘                      │
                                                            │ pygama.evt.build_tcm
                                                            ▼
                                                    ┌───────────────┐
                                                    │   tcm tier    │
                                                    │ (coincidences)│
                                                    └───────┬───────┘
                                                            │ pygama.evt.build_evt
                                                            ▼
                                                    ┌───────────────┐
                                                    │   evt tier    │
                                                    │ (event-level) │
                                                    └───────────────┘

    pygama.math and pygama.pargen support all stages above.

Related packages
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Package
     - Role
   * - `legend-pydataobj <https://legend-pydataobj.readthedocs.io>`_
     - LGDO types and LH5 I/O utilities used throughout *pygama*.
   * - `legend-daq2lh5 <https://legend-daq2lh5.readthedocs.io>`_
     - Decodes raw digitiser data into LH5 ``raw`` tier files.
   * - `dspeed <https://dspeed.readthedocs.io>`_
     - Digital signal processing framework that produces the ``dsp`` tier.
