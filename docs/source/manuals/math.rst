.. _manuals-math:

Mathematical utilities — :mod:`pygama.math`
============================================

.. currentmodule:: pygama.math

The :mod:`pygama.math` sub-package provides the statistical infrastructure
used throughout *pygama*: histogram helpers, probability distributions, and
fitting routines.  It is designed to be usable independently of the rest of
the processing chain for ad-hoc data analysis.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

:mod:`pygama.math` is organised around three concerns:

* **Histogramming** — convenience wrappers around `boost-histogram
  <https://boost-histogram.readthedocs.io>`_ that return the ``(hist, bins,
  var)`` triple expected by the fitting routines.
* **Distributions** — a library of probability density functions (PDFs) and
  cumulative distribution functions (CDFs) that model the signal and
  background shapes encountered in HPGe detector data.  All functions are JIT-
  compiled with `Numba <https://numba.readthedocs.io>`_ for speed and are
  compatible with `iminuit <https://iminuit.readthedocs.io>`_.
* **Fitting** — thin wrappers around *iminuit* that expose a consistent
  interface for binned and unbinned maximum-likelihood fits, including
  goodness-of-fit statistics and uncertainty estimation.

Submodules
----------

histogram
^^^^^^^^^

:mod:`pygama.math.histogram` provides convenience functions for working with
one-dimensional histograms.  All functions operate on the standard *pygama*
histogram triple ``(hist, bins, var)``:

* ``hist`` — array of bin counts (or weighted counts).
* ``bins`` — array of bin edges, length ``len(hist) + 1``.
* ``var`` — array of per-bin variances; defaults to ``hist`` (Poisson
  statistics) if not supplied.

Key functions:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Description
   * - :func:`~pygama.math.histogram.get_hist`
     - Bin a 1-D array of data values, returning ``(hist, bins, var)``.
       Accepts a fixed bin count, an explicit array of edges, a bin-width
       ``dx``, or any string accepted by ``numpy.histogram_bin_edges``.
   * - :func:`~pygama.math.histogram.get_fwfm`
     - Compute the full-width at a given fraction of the maximum (FWFM) of a
       histogram peak.
   * - :func:`~pygama.math.histogram.get_i_local_maxima`
     - Return the indices of local maxima above a threshold, used for
       automatic peak finding.
   * - :func:`~pygama.math.histogram.plot_hist`
     - Plot a histogram triple on a Matplotlib axes object.

.. automodule:: pygama.math.histogram
   :members:
   :undoc-members:
   :no-index:

distributions
^^^^^^^^^^^^^

:mod:`pygama.math.distributions` re-exports all distribution objects defined
in the :mod:`pygama.math.functions` sub-package.  Each distribution is a
:class:`~pygama.math.functions.pygama_continuous.PygamaContinuous` instance
(a subclass of :class:`scipy.stats.rv_continuous`) with Numba-compiled PDF
and CDF methods.

Composite distributions — a signal peak plus a background model — are the
most commonly used in HPGe peak fitting:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Distribution
     - Description
   * - :class:`~pygama.math.functions.hpge_peak.hpge_peak`
     - Full HPGe peak model: a Gaussian signal component convolved with a
       low-energy tail (exGaussian), sitting on a step-function background
       with a linear slope.  The standard model for fitting calibration peaks.
   * - :class:`~pygama.math.functions.gauss_on_step.gauss_on_step`
     - Gaussian signal on a step-function background.
   * - :class:`~pygama.math.functions.gauss_on_linear.gauss_on_linear`
     - Gaussian signal on a linear background.
   * - :class:`~pygama.math.functions.gauss_on_exponential.gauss_on_exponential`
     - Gaussian signal on an exponential background.
   * - :class:`~pygama.math.functions.gauss_on_exgauss.gauss_on_exgauss`
     - Gaussian signal on an exGaussian background.
   * - :class:`~pygama.math.functions.gauss_on_uniform.gauss_on_uniform`
     - Gaussian signal on a uniform background.
   * - :class:`~pygama.math.functions.triple_gauss_on_double_step.triple_gauss_on_double_step`
     - Three Gaussian components on a double step background; used for
       complex multi-peak regions.

Primitive distributions:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Distribution
     - Description
   * - :class:`~pygama.math.functions.gauss.gaussian`
     - Normal (Gaussian) distribution.
   * - :class:`~pygama.math.functions.exgauss.exgauss`
     - Exponentially modified Gaussian (exGaussian); models the low-energy
       tail of HPGe peaks due to incomplete charge collection.
   * - :class:`~pygama.math.functions.crystal_ball.crystal_ball`
     - Crystal Ball function; a Gaussian with a power-law low-energy tail.
   * - :class:`~pygama.math.functions.step.step`
     - Step (Heaviside-convolved-with-Gaussian) background.
   * - :class:`~pygama.math.functions.exponential.exponential`
     - Exponential background.
   * - :class:`~pygama.math.functions.linear.linear`
     - Linear background.
   * - :class:`~pygama.math.functions.uniform.uniform`
     - Uniform (flat) distribution.
   * - :class:`~pygama.math.functions.moyal.moyal`
     - Moyal distribution; approximates the Landau distribution for
       energy-loss processes.
   * - :class:`~pygama.math.functions.polynomial.nb_poly`
     - Polynomial of arbitrary degree.

.. automodule:: pygama.math.distributions
   :members:
   :undoc-members:
   :no-index:

functions
^^^^^^^^^

:mod:`pygama.math.functions` contains the individual distribution
implementations.  Each sub-module defines one distribution.

.. automodule:: pygama.math.functions
   :members:
   :undoc-members:
   :no-index:

binned_fitting
^^^^^^^^^^^^^^

:mod:`pygama.math.binned_fitting` provides :func:`~pygama.math.binned_fitting.fit_binned`,
a unified interface for fitting a function to a histogram triple.

Default behaviour is an extended binned log-likelihood fit using
:class:`iminuit.cost.ExtendedBinnedNLL`.  Least-squares and other cost
functions are also available.  The function returns arrays of best-fit
parameter values and their uncertainties, and optionally the full
:class:`~iminuit.Minuit` object for further inspection.

Additional helpers:

* :func:`~pygama.math.binned_fitting.goodness_of_fit` — compute a
  reduced chi-squared statistic and p-value for a fit result.
* :func:`~pygama.math.binned_fitting.get_bin_estimates` — evaluate a
  normalised PDF on a histogram grid for use in plotting.

.. automodule:: pygama.math.binned_fitting
   :members:
   :undoc-members:
   :no-index:

unbinned_fitting
^^^^^^^^^^^^^^^^

:mod:`pygama.math.unbinned_fitting` provides
:func:`~pygama.math.unbinned_fitting.fit_unbinned`, the unbinned counterpart
to :func:`~pygama.math.binned_fitting.fit_binned`.  The default cost function
is the extended unbinned negative log-likelihood
(:class:`iminuit.cost.ExtendedUnbinnedNLL`).

Unbinned fits are preferred when the dataset is small (fewer than a few
thousand events) or when the bin-size choice would significantly affect the
result.

.. automodule:: pygama.math.unbinned_fitting
   :members:
   :undoc-members:
   :no-index:

hpge_peak_fitting
^^^^^^^^^^^^^^^^^

:mod:`pygama.math.hpge_peak_fitting` contains higher-level routines that
combine the distribution and fitting machinery to fit the standard HPGe peak
model (:class:`~pygama.math.functions.hpge_peak.hpge_peak`) to data, with
automatic initial-parameter estimation.

.. automodule:: pygama.math.hpge_peak_fitting
   :members:
   :undoc-members:
   :no-index:

least_squares
^^^^^^^^^^^^^

:mod:`pygama.math.least_squares` provides simple linear least-squares helpers
used internally by :mod:`pygama.pargen.energy_cal`.

.. automodule:: pygama.math.least_squares
   :members:
   :undoc-members:
   :no-index:

units
^^^^^

:mod:`pygama.math.units` exposes the unit registry (backed by `pint
<https://pint.readthedocs.io>`_) and a small set of unit-conversion helpers.

.. automodule:: pygama.math.units
   :members:
   :undoc-members:
   :no-index:

utils
^^^^^

.. automodule:: pygama.math.utils
   :members:
   :undoc-members:
   :no-index:
