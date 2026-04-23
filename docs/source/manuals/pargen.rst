.. _manuals-pargen:

Parameter generation — :mod:`pygama.pargen`
============================================

.. currentmodule:: pygama.pargen

The :mod:`pygama.pargen` sub-package provides calibration and optimisation
routines that derive the numerical parameters consumed by :mod:`pygama.hit`.
Typical outputs are JSON parameter files containing calibration polynomials,
cut thresholds, and filter settings, which are then referenced in the hit-tier
configuration.

Overview
--------

HPGe detector data analysis requires several layers of calibration before
physics-quality results can be extracted:

1. **Energy calibration** — map the raw ADC-scale energy parameter to a
   physically meaningful energy scale in keV using known gamma-ray lines.
2. **Data-quality cuts** — identify and flag noise bursts, discharges, and
   other non-physics events through cuts on pulse-shape parameters.
3. **Multi-site-event (MSE) discrimination** — use the amplitude-over-energy
   ratio (A/E) or late-charge (LQ) parameter to distinguish single-site
   events (signal-like) from multi-site events (background-like).
4. **DSP filter optimisation** — tune trapezoidal-filter shaping times and
   other DSP parameters to minimise the energy resolution.
5. **Noise optimisation** — minimise the electronic noise contribution to the
   baseline energy resolution.

Each of these tasks is handled by a dedicated sub-module described below.
All calibration classes and functions make heavy use of :mod:`pygama.math`
for histogram creation, peak fitting, and survival-fraction calculation.

Submodules
----------

energy_cal
^^^^^^^^^^

:mod:`pygama.pargen.energy_cal` implements the end-to-end HPGe energy
calibration workflow.

The central class is :class:`~pygama.pargen.energy_cal.HPGeCalibration`, which
encapsulates the full calibration sequence:

1. Search the uncalibrated energy spectrum for local maxima and match them to
   known gamma-ray lines using the ratio of peak spacings as a fingerprint.
2. Refine the peak positions around the initial guesses.
3. Fit each peak with the :obj:`~pygama.math.functions.hpge_peak.hpge_peak`
   model, returning the centroid, resolution, and peak-shape parameters with
   their uncertainties.
4. Fit a polynomial (default degree 1) through the ``(ADC centroid, keV
   energy)`` pairs to obtain the calibration function.

All intermediate results are accumulated so that the full calibration history
can be inspected and stored as a JSON parameter file.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Class / Function
     - Description
   * - :class:`~pygama.pargen.energy_cal.HPGeCalibration`
     - Main calibration class; runs the full peak-finding, fitting, and
       polynomial-mapping workflow.
   * - :class:`~pygama.pargen.energy_cal.FWHMLinear`
     - Linear model for the energy dependence of the FWHM resolution.
   * - :class:`~pygama.pargen.energy_cal.FWHMQuadratic`
     - Quadratic model for the energy dependence of the FWHM resolution.

AoE_cal
^^^^^^^

:mod:`pygama.pargen.AoE_cal` calibrates the amplitude-over-energy (A/E)
parameter used to discriminate single-site events (e.g. double-beta decay
signal) from multi-site Compton-scatter backgrounds.

Key steps performed by :class:`~pygama.pargen.AoE_cal.CalAoE`:

* **Energy-dependence correction** — the A/E ratio exhibits a slow dependence
  on energy; this is modelled and corrected so that the corrected A/E is
  independent of energy.
* **Time-dependence correction** *(optional)* — long-term drifts in A/E due
  to detector ageing or temperature changes are tracked and corrected.
* **Cut determination** — the cut value is set to achieve a specified
  survival fraction in the double-escape peak (DEP), which is a proxy for
  signal-like events.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Class
     - Description
   * - :class:`~pygama.pargen.AoE_cal.CalAoE`
     - End-to-end A/E calibration class; performs energy-dependence and
       optional time-dependence corrections and determines the cut threshold.

lq_cal
^^^^^^

:mod:`pygama.pargen.lq_cal` calibrates the late-charge (LQ) parameter, which
is defined as the ratio of the charge collected after a fixed time delay to the
total charge.  LQ is sensitive to the depth of the interaction within the
detector and provides complementary MSE discrimination to A/E.

Key steps performed by :class:`~pygama.pargen.lq_cal.LQCal`:

* **Distribution fitting** — the LQ distribution is histogrammed and fitted
  with a Gaussian model in a signal-dominated region.
* **Drift-time correction** — the mean LQ value shifts with the drift time of
  charge carriers; this dependence is measured and corrected.
* **Cut determination** — the cut threshold is derived from the DEP survival
  fraction, consistent with the A/E calibration procedure.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Class
     - Description
   * - :class:`~pygama.pargen.lq_cal.LQCal`
     - End-to-end LQ calibration class with drift-time correction and cut
       determination via the DEP survival fraction.

data_cleaning
^^^^^^^^^^^^^

:mod:`pygama.pargen.data_cleaning` provides routines for calculating and
applying quality cuts that remove non-physics events from the dataset.

Typical quality parameters include the baseline level, the baseline
root-mean-square (RMS), and the current-pulse amplitude.  The module fits the
distribution of each parameter and determines cut boundaries at a specified
number of sigma.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Function
     - Description
   * - :func:`~pygama.pargen.data_cleaning.generate_cuts`
     - Fit the distribution of quality parameters and return cut boundaries.
   * - :func:`~pygama.pargen.data_cleaning.get_cut_indexes`
     - Apply a set of cut parameters to a dataset and return a boolean mask.
   * - :func:`~pygama.pargen.data_cleaning.generate_cut_classifiers`
     - Generate classifier expressions from cut parameters suitable for use
       in the hit-tier configuration.

dsp_optimize
^^^^^^^^^^^^

:mod:`pygama.pargen.dsp_optimize` implements optimisation of DSP filter
parameters (e.g. trapezoidal-filter rise time and flat-top time) to minimise
the FWHM of a calibration peak.

The :class:`~pygama.pargen.dsp_optimize.BayesianOptimizer` wraps a
Gaussian-process surrogate model to efficiently search the parameter space,
evaluating only the most promising configurations by running DSP on a
representative data sample.  A grid-search alternative is also provided via
:class:`~pygama.pargen.dsp_optimize.ParGrid`.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Class / Function
     - Description
   * - :class:`~pygama.pargen.dsp_optimize.BayesianOptimizer`
     - Gaussian-process Bayesian optimisation loop over the DSP parameter
       space; minimises a figure-of-merit (e.g. peak FWHM).
   * - :class:`~pygama.pargen.dsp_optimize.ParGrid`
     - Grid-search optimiser; exhaustively evaluates all parameter
       combinations in a user-specified grid.
   * - :func:`~pygama.pargen.dsp_optimize.run_one_dsp`
     - Run a single DSP pass on a data table and compute the figure-of-merit.
   * - :func:`~pygama.pargen.dsp_optimize.run_bayesian_optimisation`
     - Convenience function that sets up and runs the full Bayesian
       optimisation loop.

noise_optimization
^^^^^^^^^^^^^^^^^^

:mod:`pygama.pargen.noise_optimization` optimises DSP filter parameters with
the specific goal of minimising the electronic noise contribution to the
baseline energy resolution, as measured by the equivalent noise charge (ENC)
peak.

The strategy is a grid search over the filter parameter space, evaluating the
ENC peak width (FWHM) at each grid point.  A smooth interpolating spline is
then fitted to the grid to locate the global minimum.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Function
     - Description
   * - :func:`~pygama.pargen.noise_optimization.noise_optimization`
     - Run the grid search and spline fit to find the noise-minimising filter
       parameters.

survival_fractions
^^^^^^^^^^^^^^^^^^

:mod:`pygama.pargen.survival_fractions` computes signal and background
survival fractions as a function of a cut threshold.  It is used by both
:mod:`~pygama.pargen.AoE_cal` and :mod:`~pygama.pargen.lq_cal`.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Function
     - Description
   * - :func:`~pygama.pargen.survival_fractions.get_survival_fraction`
     - Compute the fraction of events in a given energy window that survive a
       cut at a specific threshold.
   * - :func:`~pygama.pargen.survival_fractions.get_sf_sweep`
     - Sweep the cut value and return the survival fraction at each point.
   * - :func:`~pygama.pargen.survival_fractions.compton_sf`
     - Estimate the survival fraction in the Compton continuum.

pz_correct
^^^^^^^^^^

:mod:`pygama.pargen.pz_correct` determines the pole-zero cancellation
constant for the HPGe preamplifier decay time, which is required by some DSP
filters.

dplms_ge_dict
^^^^^^^^^^^^^

:mod:`pygama.pargen.dplms_ge_dict` provides helpers for building the
configuration dictionaries used by the DPLMS (Data-driven Pseudo-Matched
Filter) DSP processor for germanium detectors.

For the complete parameter reference see :mod:`pygama.pargen`.
