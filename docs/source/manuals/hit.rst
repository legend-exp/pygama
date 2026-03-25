.. _manuals-hit:

Hit-tier production — :mod:`pygama.hit`
========================================

.. currentmodule:: pygama.hit

The :mod:`pygama.hit` sub-package transforms ``dsp``-tier tables into
``hit``-tier tables by evaluating user-defined column expressions.  It is the
principal mechanism through which calibrated quantities, quality-cut flags, and
other derived parameters are added to the data before event building.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

The hit tier is produced by :func:`build_hit`.  The function reads one or more
:class:`~lgdo.types.table.Table` objects from an LH5 file and, for each table,
evaluates a set of string expressions against the existing columns.  The
resulting new columns are written to an output LH5 file.

Expressions are evaluated column-by-column (not row-by-row) using
:meth:`~lgdo.types.table.Table.eval`, which internally relies on `numexpr
<https://numexpr.readthedocs.io>`_ for fast, vectorised execution without
Python overhead.

The transformation is entirely configuration-driven: no Python code is
required beyond providing the JSON configuration.  Parameters that change
between detector channels or calibration periods (e.g. calibration
coefficients) can be injected as named scalars in the configuration, keeping
the expressions readable and the parameters easily updatable.

Configuration format
--------------------

The hit configuration is a JSON object (or equivalent Python dict) with two
mandatory keys:

``outputs``
    A list of column names to write to the output file.  Only columns listed
    here appear in the ``hit`` tier; intermediate columns used only for
    subsequent expressions are discarded.

``operations``
    A mapping from output-column name to an operation descriptor.  Each
    descriptor has the following fields:

    ``expression``
        A string expression referencing existing columns by name.  Supports
        standard arithmetic operators, NumPy ufuncs available through
        ``numexpr``, and references to columns in the input table.

    ``parameters`` *(optional)*
        A mapping of parameter name to scalar value (e.g. numbers or strings)
        supported by :meth:`~lgdo.types.table.Table.eval`.  These are made
        available to the expression under their given names, allowing
        calibration constants to be stored alongside the expression without
        hard-coding them.

    ``lgdo_attrs`` *(optional)*
        A mapping of LGDO attribute name to value (e.g. ``{"units": "keV"}``),
        which is attached to the output column as metadata.

Example
^^^^^^^

The following configuration computes a calibrated energy ``calE`` from the
raw trapezoidal-filter energy ``trapEmax``, and the amplitude-over-energy
ratio ``AoE``:

.. code-block:: json

    {
      "outputs": ["calE", "AoE"],
      "operations": {
        "calE": {
          "expression": "sqrt(a + b * trapEmax**2)",
          "parameters": {"a": "1.23", "b": "42.69"},
          "lgdo_attrs": {"units": "keV"}
        },
        "AoE": {
          "expression": "A_max / calE"
        }
      }
    }

Note that ``AoE`` references ``calE``, which is itself a derived column.
Operations are evaluated in the order they are defined, so forward references
are supported.

Per-table configuration
^^^^^^^^^^^^^^^^^^^^^^^

When an LH5 file contains tables for many channels, it is often convenient to
apply slightly different configurations to different channels (e.g. different
calibration constants).  :func:`build_hit` supports this through the
``lh5_tables_config`` argument, which maps LH5 table paths to individual
configuration dictionaries::

    lh5_tables_config = {
        "ch1084803/dsp": {"outputs": [...], "operations": {...}},
        "ch1084804/dsp": {"outputs": [...], "operations": {...}},
    }

Submodule
---------

build_hit
^^^^^^^^^

.. automodule:: pygama.hit.build_hit
   :members:
   :undoc-members:
   :no-index:
