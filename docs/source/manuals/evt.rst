.. _manuals-evt:

Event building — :mod:`pygama.evt`
===================================

.. currentmodule:: pygama.evt

The :mod:`pygama.evt` sub-package is responsible for the two highest-level
steps in the LEGEND processing chain:

1. **Building the Time Coincidence Map (TCM)** — grouping hits from multiple
   channels into physics events.
2. **Building the event tier** — evaluating per-event quantities by
   aggregating channel-level data according to a user-supplied configuration.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

Data from a multi-channel detector system is stored channel-by-channel in the
``hit`` (and ``dsp``) tiers.  A single physics event will typically produce
signals in several channels at nearly the same time.  The ``evt`` module
reconstructs these physics events in two stages.

First, :func:`build_tcm` scans the per-channel timestamp columns and groups
rows that fall within a configurable coincidence window into *events*.  The
result is a TCM table containing, for each event, a list of ``(channel,
row)`` pairs — one for every hit that belongs to that event.

Second, :func:`build_evt` reads the TCM together with the ``hit`` and ``dsp``
tiers, evaluates arbitrary expressions over the contributing hits, and writes
the results to the ``evt`` tier.  The expressions and the aggregation strategy
(e.g. sum, take-first, take-last, …) are specified in a JSON configuration
file.

Submodules
----------

build_tcm
^^^^^^^^^

.. automodule:: pygama.evt.build_tcm
   :members:
   :undoc-members:
   :no-index:

build_evt
^^^^^^^^^

.. automodule:: pygama.evt.build_evt
   :members:
   :undoc-members:
   :no-index:

tcm
^^^

.. automodule:: pygama.evt.tcm
   :members:
   :undoc-members:
   :no-index:

aggregators
^^^^^^^^^^^

The :mod:`~pygama.evt.aggregators` module contains the low-level functions
that :func:`build_evt` calls to collapse per-channel hit data into a single
per-event scalar or array.  Each aggregator receives the full TCM, the list of
channels to aggregate over, an expression string, and optional query masks and
sorting columns.

Available aggregators:

* ``evaluate_to_first_or_last`` — return the expression value for the channel
  whose *sorter* column is the smallest (``is_first=True``) or largest
  (``is_first=False``) within the event (e.g. the channel with the earliest
  or latest timestamp).
* ``evaluate_to_tot_or_any`` — return the scalar sum (or logical OR) of the
  expression across all channels in the event.
* ``evaluate_to_scalar`` — evaluate a generic expression that produces one
  value per event using arbitrary aggregation logic.
* ``evaluate_at_channel`` — evaluate an expression at a specific named channel
  rather than aggregating across all channels.
* ``evaluate_to_vector`` — return the per-hit expression values as a
  :class:`~lgdo.types.vectorofvectors.VectorOfVectors`, preserving the
  per-channel structure within each event.

.. automodule:: pygama.evt.aggregators
   :members:
   :undoc-members:
   :no-index:

utils
^^^^^

.. automodule:: pygama.evt.utils
   :members:
   :undoc-members:
   :no-index:

.. _evt-modules:

modules — detector-specific processors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :mod:`pygama.evt.modules` sub-package provides ready-made *event
processors* — callables with a standardised signature that can be invoked
directly from the :func:`build_evt` JSON configuration.  Each processor
receives four positional arguments that are injected automatically by
:func:`build_evt`:

.. code-block:: python

    def my_processor(
        datainfo,       # DataInfo: tier names, file names, HDF5 groups
        tcm,            # TCMData: table_key and row_in_table arrays
        table_names,    # list[str]: hit table names for this event
        channel_mapping,  # dict | None: maps channel keys to detector names
        *,  # all following arguments are keyword-only
        arg1,
        arg2,
    ) -> lgdo.LGDO:
        pass

The processor must return an :class:`~lgdo.types.lgdo.LGDO` object (e.g. an
:class:`~lgdo.types.array.Array` or
:class:`~lgdo.types.vectorofvectors.VectorOfVectors`) suitable for insertion
into the event table.

geds
""""

:mod:`pygama.evt.modules.geds` provides processors specific to HPGe
(germanium) detectors.

* **apply_recovery_cut** — flags events that fall within a configurable time
  window after a discharge event, allowing recovery-time cuts to be applied.
* **apply_xtalk_correction** — corrects cross-talk between HPGe channels by
  subtracting the estimated cross-talk contribution from the energy of each
  hit.

.. automodule:: pygama.evt.modules.geds
   :members:
   :undoc-members:
   :no-index:

spms
""""

:mod:`pygama.evt.modules.spms` provides processors for Silicon PhotoMultiplier
(SiPM) channels, used in the liquid-argon veto system.

* **gather_pulse_data** — collects SiPM pulse observables (amplitude,
  arrival time, …) from all SiPM channels into a
  :class:`~lgdo.types.vectorofvectors.VectorOfVectors` indexed by event, with
  optional amplitude and timing masks.

.. automodule:: pygama.evt.modules.spms
   :members:
   :undoc-members:
   :no-index:

larveto
"""""""

:mod:`pygama.evt.modules.larveto` implements the statistical classifier used
for the LEGEND-200 liquid-argon veto.

* **l200_combined_test_stat** — computes a combined test statistic that
  correlates HPGe hit timing with SiPM pulse information, accounting for
  channel-specific noise rates and response curves.  The output is a per-event
  scalar suitable for use as a veto discriminant.

.. automodule:: pygama.evt.modules.larveto
   :members:
   :undoc-members:
   :no-index:

legend
""""""

:mod:`pygama.evt.modules.legend` contains higher-level convenience processors
that combine outputs from several sub-systems for the full LEGEND experimental
configuration.

.. automodule:: pygama.evt.modules.legend
   :members:
   :undoc-members:
   :no-index:

xtalk
"""""

:mod:`pygama.evt.modules.xtalk` contains the cross-talk matrix arithmetic
used by :mod:`~pygama.evt.modules.geds`.

.. automodule:: pygama.evt.modules.xtalk
   :members:
   :undoc-members:
   :no-index:

.. _evt-config:

Configuration reference
-----------------------

:func:`build_evt` is driven by a JSON (or Python dict) configuration.  A
minimal example::

    {
      "channels": {
        "geds_on": ["ch1084803", "ch1084804"],
        "spms_on": ["ch0000001", "ch0000002"]
      },
      "outputs": ["energy_sum", "is_discharge_recovery", "timestamp_first"],
      "operations": {
        "energy_sum": {
          "channels": "geds_on",
          "aggregation_mode": "sum",
          "expr": "hit.cuspEmax_ctc_cal",
          "lgdo_attrs": {"units": "keV"}
        },
        "timestamp_first": {
          "channels": "geds_on",
          "aggregation_mode": "first",
          "sort_by": "hit.timestamp",
          "expr": "hit.timestamp"
        },
        "is_discharge_recovery": {
          "channels": "geds_on",
          "aggregation_mode": "keep_at_ch:ch1084803",
          "expr": "evt.modules.geds.apply_recovery_cut",
          "kwargs": {
            "timestamps": "evt.timestamp_first",
            "flag": "hit.is_discharge",
            "time_window": 0.01
          }
        }
      }
    }

``aggregation_mode`` accepts the following values:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Mode
     - Behaviour
   * - ``first``
     - Return the expression for the channel with the smallest value of
       ``sort_by``.
   * - ``last``
     - Return the expression for the channel with the largest value of
       ``sort_by``.
   * - ``sum``
     - Scalar sum of ``expr`` across all channels.
   * - ``any``
     - Logical OR of ``expr`` across all channels.
   * - ``all``
     - Logical AND of ``expr`` across all channels.
   * - ``keep_at_ch:<name>``
     - Evaluate ``expr`` at the named channel only.
   * - ``gather``
     - Collect ``expr`` values from all channels into a
       :class:`~lgdo.types.vectorofvectors.VectorOfVectors`.
   * - ``func::<module>.<function>``
     - Call the given Python function as a custom processor (see
       :ref:`evt-modules`).
