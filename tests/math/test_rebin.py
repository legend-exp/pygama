from __future__ import annotations

import awkward as ak
import hist as bh
import numpy as np
import pytest

from pygama.math.rebin import hist_bblocks, rebin_bblocks


def _two_population_data(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(-3.0, 0.5, 2000), rng.normal(3.0, 0.5, 2000)])


def test_hist_bblocks_numpy():
    rng = np.random.default_rng(42)
    data = rng.normal(0.0, 1.0, 5000)

    h = hist_bblocks(data)

    edges = h.axes[0].edges
    assert len(edges) >= 2
    assert np.all(np.diff(edges) > 0)
    assert edges[0] <= data.min()
    assert edges[-1] >= data.max()
    assert h.values().sum() == pytest.approx(len(data))


def test_hist_bblocks_awkward_matches_numpy():
    rng = np.random.default_rng(42)
    flat = rng.normal(0.0, 1.0, 600)

    h_np = hist_bblocks(flat)

    # same data wrapped as a jagged awkward array
    jagged = ak.Array([flat[:100], flat[100:350], flat[350:]])
    h_ak = hist_bblocks(jagged)

    assert h_np.axes[0].edges == pytest.approx(h_ak.axes[0].edges)
    assert h_np.values() == pytest.approx(h_ak.values())


def test_hist_bblocks_two_populations():
    data = _two_population_data()
    h = hist_bblocks(data)

    # at least one internal change-point must fall in the gap between modes
    internal = h.axes[0].edges[1:-1]
    assert np.any((internal > -2.0) & (internal < 2.0))


def test_hist_bblocks_prebin_preserves_counts_and_finds_gap():
    data = _two_population_data()
    h = hist_bblocks(data, prebin_width=0.05)

    assert h.values().sum() == pytest.approx(len(data))
    edges = h.axes[0].edges
    assert edges[0] <= data.min()
    assert edges[-1] >= data.max()
    # gap between the two populations must still be detected
    internal = edges[1:-1]
    assert np.any((internal > -2.0) & (internal < 2.0))


def test_hist_bblocks_prebin_with_range():
    data = _two_population_data()  # spans roughly [-5, 5]
    low, high = -4.0, 4.0
    h = hist_bblocks(data, prebin_width=0.05, prebin_low=low, prebin_high=high)

    edges = h.axes[0].edges
    # output must stay within the requested range (up to ~prebin_width slack
    # on the upper edge from the ceil-and-extend step)
    assert edges[0] == low
    assert high <= edges[-1] <= high + 0.05 + 1e-9
    # only data in [low, high) contributes (as documented)
    in_range_count = int(np.sum((data >= low) & (data < high)))
    assert h.values().sum() == pytest.approx(in_range_count)


def test_hist_bblocks_prebin_low_high_require_prebin_width():
    data = _two_population_data()
    with pytest.raises(ValueError, match="prebin_width"):
        hist_bblocks(data, prebin_low=-1.0)
    with pytest.raises(ValueError, match="prebin_width"):
        hist_bblocks(data, prebin_high=1.0)


def test_hist_bblocks_prebin_width_validation():
    data = _two_population_data()
    # Test invalid prebin_width values
    with pytest.raises(ValueError, match="finite and > 0"):
        hist_bblocks(data, prebin_width=0.0)
    with pytest.raises(ValueError, match="finite and > 0"):
        hist_bblocks(data, prebin_width=-0.1)
    with pytest.raises(ValueError, match="finite and > 0"):
        hist_bblocks(data, prebin_width=np.inf)
    with pytest.raises(ValueError, match="finite and > 0"):
        hist_bblocks(data, prebin_width=np.nan)


def test_rebin_bblocks_preserves_total_counts():
    data = _two_population_data()
    h = bh.Hist(bh.axis.Regular(500, -6.0, 6.0), storage=bh.storage.Weight())
    h.fill(data)

    out = rebin_bblocks(h)

    out_var = out.variances()
    h_var = h.variances()
    assert out_var is not None
    assert h_var is not None
    assert out.values().sum() == pytest.approx(h.values().sum())
    assert out_var.sum() == pytest.approx(h_var.sum())


def test_rebin_bblocks_edges_subset_of_fine_grid():
    data = _two_population_data()
    h = bh.Hist(bh.axis.Regular(500, -6.0, 6.0), storage=bh.storage.Weight())
    h.fill(data)

    out = rebin_bblocks(h)

    fine_edges = h.axes[0].edges
    new_edges = out.axes[0].edges
    # every new edge must be exactly one of the input edges
    # (rebin_bblocks indexes fine_edges by integer position; no arithmetic)
    assert np.all(np.isin(new_edges, fine_edges))
    # endpoints must coincide with the input range
    assert new_edges[0] == fine_edges[0]
    assert new_edges[-1] == fine_edges[-1]
    # also exercise on an irregular input grid
    irregular = np.unique(
        np.concatenate([np.linspace(-6.0, 0.0, 200), np.linspace(0.0, 6.0, 350)])
    )
    h2 = bh.Hist(bh.axis.Variable(irregular), storage=bh.storage.Weight())
    h2.fill(data)
    out2 = rebin_bblocks(h2)
    new_edges2 = out2.axes[0].edges
    fine_edges2 = h2.axes[0].edges
    assert np.all(np.isin(new_edges2, fine_edges2))
    assert new_edges2[0] == fine_edges2[0]
    assert new_edges2[-1] == fine_edges2[-1]


def test_rebin_bblocks_flat_input_collapses():
    rng = np.random.default_rng(42)
    data = rng.uniform(0.0, 10.0, 10_000)
    h = bh.Hist(bh.axis.Regular(100, 0.0, 10.0), storage=bh.storage.Weight())
    h.fill(data)

    out = rebin_bblocks(h)

    # uniform input should collapse to very few blocks
    assert out.values().size <= 3
    assert out.values().sum() == pytest.approx(h.values().sum())
