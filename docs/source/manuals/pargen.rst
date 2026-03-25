.. _manuals-pargen:

Parameter generation — :mod:`pygama.pargen`
============================================

.. currentmodule:: pygama.pargen

The :mod:`pygama.pargen` sub-package provides calibration and optimisation
routines that derive the numerical parameters consumed by :mod:`pygama.hit`.
Typical outputs are JSON parameter files containing calibration polynomials,
cut thresholds, and filter settings, which are then referenced in the hit-tier
configuration.

.. contents:: Contents
   :local:
   :depth: 2

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
calibration workflow.  The central class is :class:`~pygama.pargen.energy_cal.HPGeCalibration`.

**Workflow:**

1. ``hpge_find_energy_peaks`` — searches the uncalibrated energy spectrum
   for local maxima and matches them to the known gamma-ray lines supplied
   by the user, using the ratio of peak spacings as a fingerprint.
2. ``hpge_get_energy_peaks`` — refines the peak positions around the initial
   guesses.
3. ``hpge_fit_energy_peaks`` — fits each peak with the
   :class:`~pygama.math.functions.hpge_peak.hpge_peak` model, returning
   the centroid, resolution, and peak shape parameters together with their
   uncertainties.
4. The calibration method fits a polynomial (default degree 1) through the
   ``(ADC centroid, keV energy)`` pairs to obtain the calibration function.
   The polynomial coefficients are stored in :attr:`~pygama.pargen.energy_cal.HPGeCalibration.results`.

All intermediate results are accumulated in a ``results`` dictionary so that
the full calibration history can be inspected and stored as a JSON parameter
file.

.. automodule:: pygama.pargen.energy_cal
   :members:
   :undoc-members:
   :no-index:

AoE_cal
^^^^^^^

:mod:`pygama.pargen.AoE_cal` calibrates the amplitude-over-energy (A/E)
parameter used to discriminate single-site events (e.g. double-beta decay
signal) from multi-site Compton-scatter backgrounds.

The central class is :class:`~pygama.pargen.AoE_cal.AoECalibration`.

**Key steps:**

* **Energy-dependence correction** — the A/E ratio exhibits a slow dependence
  on energy; this is modelled and corrected so that the corrected A/E is
  independent of energy.
* **Time-dependence correction** *(optional)* — long-term drifts in A/E due
  to detector ageing or temperature changes are tracked and corrected.
* **Cut determination** — the cut value is set to achieve a specified
  survival fraction in the double-escape peak (DEP), which is a proxy for
  signal-like events.
* **Survival fractions** — :mod:`~pygama.pargen.AoE_cal` wraps
  :mod:`~pygama.pargen.survival_fractions` to calculate the signal and
  background survival fractions as a function of the cut position.

.. automodule:: pygama.pargen.AoE_cal
   :members:
   :undoc-members:
   :no-index:

lq_cal
^^^^^^

:mod:`pygama.pargen.lq_cal` calibrates the late-charge (LQ) parameter, which
is defined as the ratio of the charge collected after a fixed time delay to
the total charge.  LQ is sensitive to the depth of the interaction within the
detector and provides complementary MSE discrimination to A/E.

The central class is :class:`~pygama.pargen.lq_cal.LQCal`.

**Key steps:**

* **Distribution fitting** — the LQ distribution is histogrammed and fitted
  with a Gaussian model in a signal-dominated region.
* **Drift-time correction** — the mean LQ value shifts with the drift time of
  charge carriers; this dependence is measured and corrected.
* **Cut determination** — the cut threshold is derived from the DEP survival
  fraction, consistent with the A/E calibration procedure.

.. automodule:: pygama.pargen.lq_cal
   :members:
   :undoc-members:
   :no-index:

data_cleaning
^^^^^^^^^^^^^

:mod:`pygama.pargen.data_cleaning` provides routines for calculating and
applying quality cuts that remove non-physics events from the dataset.

Typical quality parameters include the baseline level, the baseline
root-mean-square (RMS), and the current-pulse amplitude.  The module fits
the distribution of each parameter with a Gaussian model and determines
symmetric or asymmetric cut boundaries at a specified number of sigma.

Key functions:

* :func:`~pygama.pargen.data_cleaning.get_cut_parameters` — fits the
  distribution of a quality parameter and returns the cut boundaries.
* :func:`~pygama.pargen.data_cleaning.apply_cuts` — applies a set of
  cut parameters to a table and returns a boolean mask.

.. automodule:: pygama.pargen.data_cleaning
   :members:
   :undoc-members:
   :no-index:

dsp_optimize
^^^^^^^^^^^^

:mod:`pygama.pargen.dsp_optimize` implements Bayesian optimisation of DSP
filter parameters (e.g. trapezoidal-filter rise time and flat-top time) to
minimise the FWHM of a calibration peak.

The optimiser wraps a Gaussian-process regressor from `scikit-learn
<https://scikit-learn.org>`_ and uses it to build a surrogate model of the
figure-of-merit (FOM) as a function of the filter parameters.  At each
iteration, the parameter combination that maximises the expected improvement is
evaluated by running DSP on a representative data sample.

Key functions:

* :func:`~pygama.pargen.dsp_optimize.run_one_dsp` — run a single DSP
  iteration on a data table and compute the FOM.
* :func:`~pygama.pargen.dsp_optimize.optimise_energy_filter` — Bayesian
  optimisation loop over the DSP parameter space.

.. automodule:: pygama.pargen.dsp_optimize
   :members:
   :undoc-members:
   :no-index:

noise_optimization
^^^^^^^^^^^^^^^^^^

:mod:`pygama.pargen.noise_optimization` optimises DSP filter parameters with
the specific goal of minimising the electronic noise contribution to the
baseline energy resolution, as measured by the equivalent noise charge (ENC)
peak.

The strategy is a grid search over the filter parameter space, evaluating the
ENC peak width (FWHM) at each grid point.  A smooth interpolating spline is
then fitted to the grid to locate the global minimum.

.. automodule:: pygama.pargen.noise_optimization
   :members:
   :undoc-members:
   :no-index:

energy_optimisation
^^^^^^^^^^^^^^^^^^^

:mod:`pygama.pargen.energy_optimisation` provides additional helpers for
energy-resolution optimisation that complement :mod:`~pygama.pargen.dsp_optimize`.

.. automodule:: pygama.pargen.energy_optimisation
   :members:
   :undoc-members:
   :no-index:

survival_fractions
^^^^^^^^^^^^^^^^^^

:mod:`pygama.pargen.survival_fractions` computes signal and background
survival fractions as a function of a cut threshold.  It is used by both
:mod:`~pygama.pargen.AoE_cal` and :mod:`~pygama.pargen.lq_cal`.

Key functions:

* :func:`~pygama.pargen.survival_fractions.get_survival_fraction` — compute
  the fraction of events in a given energy window that survive a cut.
* :func:`~pygama.pargen.survival_fractions.get_sf_sweep` — sweep the cut
  value and return the survival fraction at each point.
* :func:`~pygama.pargen.survival_fractions.compton_sf` — estimate the
  survival fraction in the Compton continuum.

.. automodule:: pygama.pargen.survival_fractions
   :members:
   :undoc-members:
   :no-index:

pz_correct
^^^^^^^^^^

:mod:`pygama.pargen.pz_correct` determines the pole-zero cancellation
constant for the HPGe preamplifier decay time, which is required by some DSP
filters.

.. automodule:: pygama.pargen.pz_correct
   :members:
   :undoc-members:
   :no-index:

dplms_ge_dict
^^^^^^^^^^^^^

:mod:`pygama.pargen.dplms_ge_dict` provides helpers for building the
configuration dictionaries used by the DPLMS (Data-driven Pseudo-Matched
Filter) DSP processor for germanium detectors.

.. automodule:: pygama.pargen.dplms_ge_dict
   :members:
   :undoc-members:
   :no-index:

utils
^^^^^

.. automodule:: pygama.pargen.utils
   :members:
   :undoc-members:
   :no-index:
