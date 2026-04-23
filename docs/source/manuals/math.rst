.. _manuals-math:

Mathematical utilities — :mod:`pygama.math`
============================================

.. currentmodule:: pygama.math

The :mod:`pygama.math` sub-package provides the statistical infrastructure
used throughout *pygama*: histogram helpers, probability distributions, and
fitting routines.  It is designed to be usable independently of the rest of
the processing chain for ad-hoc data analysis.

Overview
--------

:mod:`pygama.math` is organised around three concerns:

* **Histogramming** — convenience wrappers around `boost-histogram
  <https://boost-histogram.readthedocs.io>`_ that return the ``(hist, bins,
  var)`` triple expected by the fitting routines.
* **Distributions** — a library of probability density functions (PDFs) and
  cumulative distribution functions (CDFs) that model the signal and
  background shapes encountered in HPGe detector data.  All functions are
  JIT-compiled with `Numba <https://numba.readthedocs.io>`_ for speed and are
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

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Function
     - Description
   * - :func:`~pygama.math.histogram.get_hist`
     - Bin a 1-D array of data values, returning ``(hist, bins, var)``.
       Accepts a fixed bin count, an explicit array of edges, a bin-width
       ``dx``, or any string accepted by ``numpy.histogram_bin_edges``.
   * - :func:`~pygama.math.histogram.get_fwhm`
     - Compute the full-width at half-maximum (FWHM) of a histogram peak.
   * - :func:`~pygama.math.histogram.get_fwfm`
     - Compute the full-width at a given fraction of the maximum (FWFM).
   * - :func:`~pygama.math.histogram.get_i_local_maxima`
     - Return the indices of local maxima above a threshold, used for
       automatic peak finding.
   * - :func:`~pygama.math.histogram.get_bin_estimates`
     - Evaluate a normalised PDF on a histogram grid for use in plotting.
   * - :func:`~pygama.math.histogram.plot_hist`
     - Plot a histogram triple on a Matplotlib axes object.

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
   :widths: 40 60

   * - Distribution
     - Description
   * - :obj:`~pygama.math.functions.hpge_peak.hpge_peak`
     - Full HPGe peak model: a Gaussian signal component convolved with a
       low-energy tail (exGaussian), sitting on a step-function background
       with a linear slope.  The standard model for fitting calibration peaks.
   * - :obj:`~pygama.math.functions.gauss_on_step.gauss_on_step`
     - Gaussian signal on a step-function background.
   * - :obj:`~pygama.math.functions.gauss_on_linear.gauss_on_linear`
     - Gaussian signal on a linear background.
   * - :obj:`~pygama.math.functions.gauss_on_exponential.gauss_on_exponential`
     - Gaussian signal on an exponential background.
   * - :obj:`~pygama.math.functions.gauss_on_exgauss.gauss_on_exgauss`
     - Gaussian signal on an exGaussian background.
   * - :obj:`~pygama.math.functions.gauss_on_uniform.gauss_on_uniform`
     - Gaussian signal on a uniform background.
   * - :obj:`~pygama.math.functions.triple_gauss_on_double_step.triple_gauss_on_double_step`
     - Three Gaussian components on a double step background; used for
       complex multi-peak regions.

Primitive distributions:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Distribution
     - Description
   * - :obj:`~pygama.math.functions.gauss.gaussian`
     - Normal (Gaussian) distribution.
   * - :obj:`~pygama.math.functions.exgauss.exgauss`
     - Exponentially modified Gaussian; models the low-energy tail of HPGe
       peaks due to incomplete charge collection.
   * - :obj:`~pygama.math.functions.crystal_ball.crystal_ball`
     - Crystal Ball function; a Gaussian with a power-law low-energy tail.
   * - :obj:`~pygama.math.functions.step.step`
     - Step (Heaviside-convolved-with-Gaussian) background.
   * - :obj:`~pygama.math.functions.exponential.exponential`
     - Exponential background.
   * - :obj:`~pygama.math.functions.linear.linear`
     - Linear background.
   * - :obj:`~pygama.math.functions.uniform.uniform`
     - Uniform (flat) distribution.
   * - :obj:`~pygama.math.functions.moyal.moyal`
     - Moyal distribution; approximates the Landau distribution for
       energy-loss processes.
   * - :obj:`~pygama.math.functions.polynomial.nb_poly`
     - Polynomial of arbitrary degree.

binned_fitting
^^^^^^^^^^^^^^

:mod:`pygama.math.binned_fitting` provides a unified interface for fitting a
function to a histogram triple.  The default cost function is an extended
binned log-likelihood fit (:class:`iminuit.cost.ExtendedBinnedNLL`).

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Function
     - Description
   * - :func:`~pygama.math.binned_fitting.fit_binned`
     - Fit a model to a histogram triple; returns best-fit parameter values,
       uncertainties, and optionally the full :class:`~iminuit.Minuit` object.
   * - :func:`~pygama.math.binned_fitting.goodness_of_fit`
     - Compute a reduced chi-squared statistic and p-value for a fit result.
   * - :func:`~pygama.math.binned_fitting.gauss_mode_width_max`
     - Fit a Gaussian to a histogram peak and return the mode, width, and
       maximum, useful for robust initial-parameter estimation.

unbinned_fitting
^^^^^^^^^^^^^^^^

:mod:`pygama.math.unbinned_fitting` provides the unbinned counterpart to
:mod:`~pygama.math.binned_fitting`.  The default cost function is the extended
unbinned negative log-likelihood (:class:`iminuit.cost.ExtendedUnbinnedNLL`).

Unbinned fits are preferred when the dataset is small (fewer than a few
thousand events) or when the bin-size choice would significantly affect the
result.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Function
     - Description
   * - :func:`~pygama.math.unbinned_fitting.fit_unbinned`
     - Fit a model to an array of unbinned data values using extended
       unbinned negative log-likelihood minimisation.

hpge_peak_fitting
^^^^^^^^^^^^^^^^^

:mod:`pygama.math.hpge_peak_fitting` contains higher-level routines that
combine the distribution and fitting machinery to fit the standard HPGe peak
model (:obj:`~pygama.math.functions.hpge_peak.hpge_peak`) to data, with
automatic initial-parameter estimation.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Function
     - Description
   * - :func:`~pygama.math.hpge_peak_fitting.hpge_peak_fwhm`
     - Compute the FWHM of an HPGe peak from the fit parameters, propagating
       uncertainties analytically.
   * - :func:`~pygama.math.hpge_peak_fitting.hpge_peak_fwfm`
     - Compute the full-width at a given fraction of the maximum.
   * - :func:`~pygama.math.hpge_peak_fitting.hpge_peak_mode`
     - Return the mode (peak position) of the HPGe peak model.

least_squares
^^^^^^^^^^^^^

:mod:`pygama.math.least_squares` provides simple linear least-squares helpers
used internally by :mod:`pygama.pargen.energy_cal`.

units
^^^^^

:mod:`pygama.math.units` exposes the unit registry (backed by `pint
<https://pint.readthedocs.io>`_) and a small set of unit-conversion helpers.

For the complete parameter reference see :mod:`pygama.math`.
