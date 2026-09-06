from __future__ import annotations

import logging
import types

import numpy as np
import scipy

import pygama.math.distributions as pgd
from pygama.pargen import energy_optimisation
from pygama.pargen.energy_optimisation import (
    _fit_at_limit,
    _fit_bounds_by_index,
    get_peak_fwhm_with_dt_corr,
)


def test_import():
    pass


def test_scipy_version():
    assert scipy.__version__ != ""
    assert scipy.__version__ is not None


def _stub_fit(**limits):
    req = [str(a) for a in pgd.hpge_peak.required_args()]
    return types.SimpleNamespace(
        parameters=req,
        limits=[limits.get(name, (-np.inf, np.inf)) for name in req],
    )


def test_fit_bounds_by_index_matches_the_fit_limits():
    pars = [-25.0, 25.0, 8000, 0.0, 1.0, 0.05, 5.0, 800, 0.0]
    req = [str(a) for a in pgd.hpge_peak.required_args()]
    fit = _stub_fit(
        htail=(0.0, 0.5), n_sig=(0.0, np.inf), sigma=(0.0, 25.0), tau=(0.19, 9.7)
    )

    bounds = _fit_bounds_by_index(pgd.hpge_peak, pars, fit)

    # htail is limited to (0, 0.5) by the fit, not the [0, 1] the fwfm
    # evaluator happens to accept
    assert bounds[req.index("htail")] == (0.0, 0.5)
    # an infinite side becomes None (open), not inf
    assert bounds[req.index("n_sig")] == (0.0, None)
    assert bounds[req.index("tau")] == (0.19, 9.7)
    # x_lo / x_hi are unbounded on both sides and must not appear
    assert req.index("x_lo") not in bounds
    assert req.index("x_hi") not in bounds


def test_fit_bounds_by_index_drops_bounds_excluding_the_fitted_point():
    """A limit that excludes its own fitted value would reject every draw.

    ``tau`` is limited by the *guess* sigma, so a range recomputed from the
    fitted sigma can sit below the fitted tau; such a bound must be discarded
    rather than applied.
    """
    req = [str(a) for a in pgd.hpge_peak.required_args()]
    pars = [-25.0, 25.0, 8000, 0.0, 1.0, 0.02, 5.65, 800, 0.0]
    fit = _stub_fit(htail=(0.0, 0.5), tau=(0.0999, 4.996))  # excludes tau=5.65

    bounds = _fit_bounds_by_index(pgd.hpge_peak, pars, fit)

    assert req.index("tau") not in bounds
    assert bounds[req.index("htail")] == (0.0, 0.5)


def test_fit_bounds_by_index_returns_none_without_a_fit():
    assert _fit_bounds_by_index(pgd.hpge_peak, [1, 2, 3], None) is None
    assert _fit_bounds_by_index(object(), [1, 2, 3], _stub_fit()) is None


def test_fit_at_limit():
    assert _fit_at_limit(None) is False
    assert _fit_at_limit(object()) is False

    at_limit = types.SimpleNamespace(
        fmin=types.SimpleNamespace(has_parameters_at_limit=True)
    )
    assert _fit_at_limit(at_limit) is True

    interior = types.SimpleNamespace(
        fmin=types.SimpleNamespace(has_parameters_at_limit=False)
    )
    assert _fit_at_limit(interior) is False


def _toy_peak(htail, n_sig=6000, seed=3):
    """Draw a toy hpge peak by rejection sampling from its own pdf."""
    x_lo, x_hi = -25.0, 25.0
    pars = [x_lo, x_hi, n_sig, 0.0, 1.0, htail, 5.0, 600, 0.0]
    rng = np.random.default_rng(seed)
    grid = np.linspace(x_lo, x_hi, 2001)
    _, dens = pgd.hpge_peak.pdf_ext(grid, *pars)
    pmax = dens.max()
    out = []
    while len(out) < n_sig:
        cand = rng.uniform(x_lo, x_hi, size=4 * n_sig)
        keep = rng.uniform(0, pmax, size=4 * n_sig)
        _, dvals = pgd.hpge_peak.pdf_ext(cand, *pars)
        out.extend(cand[keep < dvals].tolist())
    return np.array(out[:n_sig])


def test_get_peak_fwhm_with_dt_corr_returns_finite_errors(caplog):
    """End-to-end: the bootstrap must produce a usable error for a normal peak."""
    energies = _toy_peak(htail=0.05) + 2039.0
    dt = np.zeros_like(energies)

    with caplog.at_level(logging.WARNING):
        fwhm, _fwhm_o_max, fwhm_err, fwhm_o_max_err, _chisqr, _n_sig, *_ = (
            get_peak_fwhm_with_dt_corr(
                energies,
                0.0,
                dt,
                pgd.hpge_peak,
                peak=2039.0,
                kev_width=(20, 20),
                bin_width=0.5,
            )
        )

    assert np.isfinite(fwhm)
    assert np.isfinite(fwhm_err)
    assert fwhm_err > 0
    assert np.isfinite(fwhm_o_max_err)
    assert fwhm_o_max_err > 0
    # the bootstrap must fill its sample: an over-tight bound that excluded the
    # fitted point would starve it and silently degrade the error to nan
    assert "draws valid" not in caplog.text


def test_boundary_fit_is_flagged(caplog):
    """A fit sitting on a limit must warn that its uncertainties are unreliable."""
    energies = _toy_peak(htail=0.05) + 2039.0
    dt = np.zeros_like(energies)

    real_fit = energy_optimisation.pgc.unbinned_staged_energy_fit

    def fit_at_limit(*args, **kwargs):
        result = list(real_fit(*args, **kwargs))
        result[7] = types.SimpleNamespace(
            fmin=types.SimpleNamespace(has_parameters_at_limit=True)
        )
        return tuple(result)

    energy_optimisation.pgc.unbinned_staged_energy_fit = fit_at_limit
    try:
        with caplog.at_level(logging.WARNING):
            get_peak_fwhm_with_dt_corr(
                energies,
                0.0,
                dt,
                pgd.hpge_peak,
                peak=2039.0,
                kev_width=(20, 20),
                bin_width=0.5,
            )
    finally:
        energy_optimisation.pgc.unbinned_staged_energy_fit = real_fit

    assert "parameters at a limit" in caplog.text
