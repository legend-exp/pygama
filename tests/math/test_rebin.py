from __future__ import annotations

import awkward as ak
import hist as bh
import numpy as np
import pytest

from pygama.math.rebin import (
    _collapse_peaks,
    _find_peak_regions,
    _uniform_edges_with_peaks,
    hist_bblocks,
    hist_bblocks_with_peaks,
    hist_uniform_with_peaks,
    rebin_bblocks,
    rebin_bblocks_with_peaks,
    rebin_uniform_with_peaks,
)


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


def _hist_from_arrays(edges: np.ndarray, values: np.ndarray) -> bh.Hist:
    """Build a Variable-axis Weight-storage histogram with Poisson variances."""
    h = bh.Hist(bh.axis.Variable(edges), storage=bh.storage.Weight())
    view = np.asarray(h.view())
    view["value"] = values
    view["variance"] = values
    return h


def _hist_from_widths_rates(blocks: list[tuple[float, float]]) -> bh.Hist:
    """Build a histogram from a list of ``(width, rate)`` blocks.

    Counts are deterministic ``width * rate`` (no Poisson noise) so the
    rate shape on each side of a peak is exactly under test.
    """
    widths = np.array([w for w, _ in blocks])
    rates = np.array([r for _, r in blocks])
    edges = np.concatenate([[0.0], np.cumsum(widths)])
    counts = widths * rates
    return _hist_from_arrays(edges, counts)


def test_collapse_peaks_merges_descending_peak_and_stops_at_width_jump():
    # narrow bins forming a peak (rates rise then fall), bordered on each
    # side by a wide continuum bin. Walk should include the narrow run
    # only, stopping at the width jump to continuum.
    # widths: [10, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 10]
    # rates:  [100, 200, 1000, 5000, 50000, 5000, 1000, 200, 100]
    h = _hist_from_widths_rates(
        [
            (10.0, 100.0),  # left continuum
            (0.5, 200.0),  # peak structure starts
            (0.5, 1000.0),
            (0.5, 5000.0),
            (0.5, 50000.0),  # peak top
            (0.5, 5000.0),
            (0.5, 1000.0),
            (0.5, 200.0),  # peak structure ends
            (10.0, 100.0),  # right continuum
        ]
    )
    out = _collapse_peaks(h, n_sigma=5.0, width_factor=5.0)

    # 3 output bins: left continuum, merged peak, right continuum
    new_edges = out.axes[0].edges
    assert out.values().size == 3
    in_edges = h.axes[0].edges
    assert new_edges[0] == in_edges[0]
    assert new_edges[1] == in_edges[1]  # boundary between continuum and peak
    assert new_edges[2] == in_edges[-2]  # boundary between peak and continuum
    assert new_edges[3] == in_edges[-1]


def test_collapse_peaks_walk_stops_at_rate_local_minimum():
    # two peaks separated by a narrow valley (all narrow bins; no width jump
    # between them). Walk must stop at the rate minimum, giving two regions.
    h = _hist_from_widths_rates(
        [
            (5.0, 100.0),  # left continuum
            (0.5, 1000.0),
            (0.5, 5000.0),
            (0.5, 30000.0),  # peak 1
            (0.5, 5000.0),
            (0.5, 2000.0),  # valley between peaks
            (0.5, 5000.0),
            (0.5, 30000.0),  # peak 2
            (0.5, 5000.0),
            (0.5, 1000.0),
            (5.0, 100.0),  # right continuum
        ]
    )
    out = _collapse_peaks(h, n_sigma=5.0, width_factor=5.0)

    new_edges = out.axes[0].edges
    new_widths = np.diff(new_edges)
    in_edges = h.axes[0].edges
    # expect 5 output bins: left continuum (5), peak1 merge, valley (0.5),
    # peak2 merge, right continuum (5)
    assert out.values().size == 5
    # exactly two merged peak regions of width 2.0 (4 narrow bins of 0.5)
    assert int(np.sum(np.isclose(new_widths, 2.0))) == 2
    # the valley bin stays as a single 0.5 bin
    assert int(np.sum(np.isclose(new_widths, 0.5))) == 1
    # verify the valley bin's identity: input bin index 5 (zero-indexed)
    # is the valley (rate 2000). Its left/right edges are in_edges[5]/[6].
    # The output bin at index 2 must coincide with that valley bin.
    assert np.isclose(new_edges[2], in_edges[5])
    assert np.isclose(new_edges[3], in_edges[6])


def test_collapse_peaks_walk_stops_at_width_jump_even_if_rate_descends():
    # rate keeps descending past the peak/continuum boundary, but the
    # width jumps up — walk must stop at the jump, not extend into the
    # continuum bin.
    h = _hist_from_widths_rates(
        [
            (10.0, 30.0),  # far left continuum
            (0.5, 100.0),  # rising side
            (0.5, 1000.0),
            (0.5, 10000.0),  # peak
            (0.5, 1000.0),
            (0.5, 100.0),  # last narrow bin, rate still > continuum
            (10.0, 50.0),  # continuum bin: lower rate than 100 → pure
            # rate-descent would walk INTO it
        ]
    )
    out = _collapse_peaks(h, n_sigma=5.0, width_factor=5.0)
    out_edges = out.axes[0].edges
    out_widths = np.diff(out_edges)
    # the wide continuum bins must remain in the output as 10 keV bins
    assert int(np.sum(np.isclose(out_widths, 10.0))) == 2


def test_collapse_peaks_preserves_counts_and_variances():
    h = _hist_from_widths_rates(
        [
            (5.0, 100.0),
            (0.5, 500.0),
            (0.5, 5000.0),
            (0.5, 50000.0),
            (0.5, 5000.0),
            (0.5, 500.0),
            (5.0, 100.0),
            (0.5, 500.0),
            (0.5, 8000.0),
            (0.5, 80000.0),
            (0.5, 8000.0),
            (0.5, 500.0),
            (5.0, 100.0),
        ]
    )
    out = _collapse_peaks(h, n_sigma=5.0)

    h_var = h.variances()
    out_var = out.variances()
    assert h_var is not None
    assert out_var is not None
    assert out.values().sum() == pytest.approx(h.values().sum())
    assert out_var.sum() == pytest.approx(h_var.sum())


def test_collapse_peaks_edges_subset_of_input():
    h = _hist_from_widths_rates(
        [
            (10.0, 100.0),
            (0.5, 1000.0),
            (0.5, 50000.0),
            (0.5, 1000.0),
            (10.0, 100.0),
        ]
    )
    out = _collapse_peaks(h)
    fine_edges = h.axes[0].edges
    new_edges = out.axes[0].edges
    assert np.all(np.isin(new_edges, fine_edges))
    assert new_edges[0] == fine_edges[0]
    assert new_edges[-1] == fine_edges[-1]


def test_collapse_peaks_no_peaks_returns_unchanged():
    # flat rate everywhere — no rate-local-maxima, no peaks detected.
    h = _hist_from_widths_rates([(1.0, 100.0)] * 10)
    out = _collapse_peaks(h, n_sigma=5.0)
    assert np.array_equal(out.axes[0].edges, h.axes[0].edges)
    assert np.array_equal(out.values(), h.values())


def test_collapse_peaks_max_bin_width_caps_wide_peak_walk():
    # a peak on a 3-keV bin, surrounded by gradually widening shoulders
    # that would all sit under the peak-relative cap (3 * 3 = 9 keV).
    # The absolute cap must clip the walk before reaching the wide bins.
    h = _hist_from_widths_rates(
        [
            (50.0, 30.0),  # far-left continuum
            (3.0, 8500.0),  # rising shoulder
            (3.0, 11000.0),  # peak
            (2.0, 10000.0),  # shoulder
            (
                8.5,
                9000.0,
            ),  # would pass relative cap (8.5 < 9) — should be stopped by absolute cap
            (5.0, 8500.0),
            (50.0, 30.0),
        ]
    )

    # without absolute cap: the walk drags in the 8.5-keV bin
    out_loose = _collapse_peaks(h, n_sigma=2.0, width_factor=3.0)
    widths_loose = np.diff(out_loose.axes[0].edges)
    assert widths_loose.max() > 10.0  # merged region grew through the 8.5 bin

    # with absolute cap of 5 keV: walk stops before the 8.5-keV bin
    out_tight = _collapse_peaks(h, n_sigma=2.0, width_factor=3.0, max_bin_width=5.0)
    widths_tight = np.diff(out_tight.axes[0].edges)
    # the wide bins remain at their original sizes (8.5 and 50)
    assert int(np.sum(np.isclose(widths_tight, 8.5))) == 1
    assert int(np.sum(np.isclose(widths_tight, 50.0))) == 2


def test_collapse_peaks_handles_low_stats_peak_on_wider_bin():
    # peak sits on a 3 keV bin (low-stats peak); neighbors are also moderately
    # wide. width_factor only kicks in at the boundary to wide continuum.
    h = _hist_from_widths_rates(
        [
            (50.0, 30.0),  # far-left continuum
            (3.0, 100.0),  # peak shoulder
            (3.0, 500.0),  # peak top
            (3.0, 100.0),  # peak shoulder
            (50.0, 30.0),  # far-right continuum
        ]
    )
    out = _collapse_peaks(h, n_sigma=5.0, width_factor=5.0)
    new_edges = out.axes[0].edges
    new_widths = np.diff(new_edges)
    # all three 3-keV bins should be merged into one 9-keV bin
    assert int(np.sum(np.isclose(new_widths, 9.0))) == 1
    # continuum bins preserved
    assert int(np.sum(np.isclose(new_widths, 50.0))) == 2


def test_collapse_peaks_with_rebin_bblocks_end_to_end():
    rng = np.random.default_rng(42)
    data = np.concatenate(
        [
            rng.uniform(0.0, 100.0, 20_000),
            rng.normal(50.0, 0.3, 10_000),
        ]
    )
    fine = bh.Hist(bh.axis.Regular(2000, 0.0, 100.0), storage=bh.storage.Weight())
    fine.fill(data)

    bb = rebin_bblocks(fine)
    out = _collapse_peaks(bb, n_sigma=5.0)

    assert out.values().sum() == pytest.approx(bb.values().sum())
    assert out.values().size <= bb.values().size
    # there should be a wide merged output bin near the true peak at 50
    new_edges = out.axes[0].edges
    new_widths = np.diff(new_edges)
    # at least one merged output bin centered close to 50
    for i, w in enumerate(new_widths):
        if w > 0.5:  # a region that combined multiple fine bblocks bins
            center = (new_edges[i] + new_edges[i + 1]) / 2
            if 45.0 < center < 55.0:
                return  # success
    pytest.fail("no merged region found around the true peak at 50")


def test_collapse_peaks_rejects_tail_false_positives_from_width_variation():
    # narrow bins with alternating widths along a monotonically rising rate.
    # counts have local maxima at the wider bins, but those are not
    # rate-local-maxima and must be rejected by the rate filter.
    widths_arr = np.array([0.1, 0.2, 0.1, 0.2, 0.1])
    edges = np.concatenate([[0.0], np.cumsum(widths_arr)])
    rates = np.array([100.0, 110.0, 120.0, 130.0, 140.0]) * 1000.0
    counts = widths_arr * rates
    h = _hist_from_arrays(edges, counts)

    out = _collapse_peaks(h, n_sigma=2.0)
    assert np.array_equal(out.axes[0].edges, edges)
    assert np.array_equal(out.values(), counts)


def test_collapse_peaks_single_bin_histogram():
    # one-bin histogram: no local maxima possible, output equals input
    edges = np.array([0.0, 1.0])
    h = _hist_from_arrays(edges, np.array([42.0]))
    out = _collapse_peaks(h)
    assert np.array_equal(out.axes[0].edges, edges)
    assert np.array_equal(out.values(), h.values())


def test_collapse_peaks_all_zero_histogram():
    # all-zero histogram: rate is all zeros, no peaks, output unchanged
    edges = np.linspace(0.0, 10.0, 11)
    h = _hist_from_arrays(edges, np.zeros(10))
    out = _collapse_peaks(h)
    assert np.array_equal(out.axes[0].edges, edges)
    assert np.array_equal(out.values(), h.values())


def test_collapse_peaks_spike_at_boundary_is_passed_through():
    # scipy.signal.find_peaks never reports indices 0 or n-1 as peaks,
    # so a spike at the very first or last bin is silently passed through
    # unchanged (not a bug, but a documented behavior).
    edges = np.linspace(0.0, 10.0, 11)
    values = np.array([100.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    h = _hist_from_arrays(edges, values)
    out = _collapse_peaks(h, n_sigma=5.0)
    # output is unchanged (no peak detected)
    assert np.array_equal(out.axes[0].edges, edges)
    assert np.array_equal(out.values(), values)


def test_collapse_peaks_rejects_negative_counts():
    edges = np.linspace(0.0, 10.0, 11)
    h = bh.Hist(bh.axis.Variable(edges), storage=bh.storage.Weight())
    view = np.asarray(h.view())
    view["value"] = [1.0, 2.0, 3.0, 4.0, 5.0, -1.0, 5.0, 4.0, 3.0, 2.0]
    with pytest.raises(ValueError, match="non-negative"):
        _collapse_peaks(h)


def test_collapse_peaks_adjacent_regions_not_merged():
    # two close peaks where each walk stops exactly one bin apart;
    # the regions are adjacent (share an edge) but do NOT overlap, and
    # must remain two separate output bins, preserving the boundary.
    h = _hist_from_widths_rates(
        [
            (5.0, 100.0),  # left continuum
            (0.5, 20000.0),  # peak 1 (sharp, isolated bin)
            (0.5, 1000.0),  # immediate valley
            (0.5, 20000.0),  # peak 2 (sharp, isolated bin)
            (5.0, 100.0),  # right continuum
        ]
    )
    out = _collapse_peaks(h, n_sigma=5.0, width_factor=5.0)
    # output: 5 bins (left cont, peak1, valley, peak2, right cont) —
    # the two peaks must NOT merge even though their walks could end
    # at adjacent indices
    assert out.values().size == 5


def test_collapse_peaks_input_validation():
    edges = np.linspace(0.0, 100.0, 101)
    h = bh.Hist(bh.axis.Variable(edges), storage=bh.storage.Weight())

    with pytest.raises(ValueError, match="n_sigma"):
        _collapse_peaks(h, n_sigma=0.0)
    with pytest.raises(ValueError, match="n_sigma"):
        _collapse_peaks(h, n_sigma=-1.0)
    with pytest.raises(ValueError, match="width_factor"):
        _collapse_peaks(h, width_factor=1.0)
    with pytest.raises(ValueError, match="width_factor"):
        _collapse_peaks(h, width_factor=0.5)
    with pytest.raises(ValueError, match="max_bin_width"):
        _collapse_peaks(h, max_bin_width=0.0)
    with pytest.raises(ValueError, match="max_bin_width"):
        _collapse_peaks(h, max_bin_width=-1.0)

    h2 = bh.Hist(
        bh.axis.Regular(10, 0.0, 1.0),
        bh.axis.Regular(10, 0.0, 1.0),
        storage=bh.storage.Weight(),
    )
    with pytest.raises(ValueError, match="1D"):
        _collapse_peaks(h2)


# ---------------------------------------------------------------------------
# _find_peak_regions
# ---------------------------------------------------------------------------


def test_find_peak_regions_no_peaks_returns_empty():
    h = _hist_from_widths_rates([(1.0, 100.0)] * 10)
    regions = _find_peak_regions(h, n_sigma=5.0)
    assert regions.shape == (0, 2)


def test_find_peak_regions_returns_edges_in_axis_units():
    h = _hist_from_widths_rates(
        [
            (10.0, 100.0),
            (0.5, 1000.0),
            (0.5, 50000.0),
            (0.5, 1000.0),
            (10.0, 100.0),
        ]
    )
    regions = _find_peak_regions(h, n_sigma=5.0, width_factor=5.0)
    assert regions.shape == (1, 2)
    # the merged peak should span the three narrow bins (indices 1-3)
    in_edges = h.axes[0].edges
    assert np.isclose(regions[0, 0], in_edges[1])
    assert np.isclose(regions[0, 1], in_edges[4])


# ---------------------------------------------------------------------------
# _uniform_edges_with_peaks
# ---------------------------------------------------------------------------


def test_uniform_edges_with_peaks_no_peaks_returns_uniform():
    edges = _uniform_edges_with_peaks(
        0.0, 10.0, bin_width=1.0, peak_regions=np.empty((0, 2))
    )
    assert np.allclose(edges, np.arange(11.0))


def test_uniform_edges_with_peaks_inserts_peaks_and_drops_interior_uniform():
    # peak region [3.2, 4.7] should drop uniform edge 4 and insert 3.2, 4.7
    regions = np.array([[3.2, 4.7]])
    edges = _uniform_edges_with_peaks(0.0, 10.0, bin_width=1.0, peak_regions=regions)
    # 4 was strictly inside the peak region, so it must be gone
    assert 4.0 not in edges
    # peak boundaries must be present
    assert 3.2 in edges
    assert 4.7 in edges
    # uniform edges outside the peak region must survive
    for x in [0.0, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
        assert x in edges


def test_uniform_edges_with_peaks_bin_width_larger_than_range_keeps_two_edges():
    edges = _uniform_edges_with_peaks(
        0.0, 1.0, bin_width=10.0, peak_regions=np.empty((0, 2))
    )
    assert edges[0] == 0.0
    assert edges[-1] == 1.0


def test_uniform_edges_with_peaks_validation():
    with pytest.raises(ValueError, match="bin_width"):
        _uniform_edges_with_peaks(
            0.0, 1.0, bin_width=0.0, peak_regions=np.empty((0, 2))
        )
    with pytest.raises(ValueError, match="bin_width"):
        _uniform_edges_with_peaks(
            0.0, 1.0, bin_width=-1.0, peak_regions=np.empty((0, 2))
        )
    with pytest.raises(ValueError, match="greater than"):
        _uniform_edges_with_peaks(
            1.0, 0.0, bin_width=0.1, peak_regions=np.empty((0, 2))
        )


# ---------------------------------------------------------------------------
# hist_bblocks_with_peaks / rebin_bblocks_with_peaks
# ---------------------------------------------------------------------------


def _peaked_data(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [
            rng.uniform(0.0, 100.0, 20_000),
            rng.normal(50.0, 0.3, 10_000),
        ]
    )


def test_hist_bblocks_with_peaks_preserves_total_counts():
    data = _peaked_data()
    out = hist_bblocks_with_peaks(data, prebin_width=0.05, n_sigma=5.0)
    assert out.values().sum() == pytest.approx(len(data))


def test_hist_bblocks_with_peaks_consolidates_peak_region():
    # the function should produce a merged bin around the true peak at 50
    data = _peaked_data()
    out = hist_bblocks_with_peaks(data, prebin_width=0.05, n_sigma=5.0)
    edges = out.axes[0].edges
    widths = np.diff(edges)
    # at least one wide bin should contain the peak center
    for i, w in enumerate(widths):
        center = (edges[i] + edges[i + 1]) / 2
        if w > 0.5 and 45.0 < center < 55.0:
            return  # success
    pytest.fail("no merged region found around the true peak at 50")


def test_rebin_bblocks_with_peaks_equals_collapse_after_rebin():
    data = _peaked_data()
    fine = bh.Hist(bh.axis.Regular(2000, 0.0, 100.0), storage=bh.storage.Weight())
    fine.fill(data)

    expected = _collapse_peaks(rebin_bblocks(fine, p0=0.05), n_sigma=5.0)
    actual = rebin_bblocks_with_peaks(fine, n_sigma=5.0, p0=0.05)

    assert np.allclose(expected.axes[0].edges, actual.axes[0].edges)
    assert np.allclose(expected.values(), actual.values())


# ---------------------------------------------------------------------------
# hist_uniform_with_peaks / rebin_uniform_with_peaks
# ---------------------------------------------------------------------------


def test_hist_uniform_with_peaks_continuum_is_bin_width():
    data = _peaked_data()
    out = hist_uniform_with_peaks(data, bin_width=1.0, prebin_width=0.05, n_sigma=5.0)
    edges = out.axes[0].edges
    widths = np.diff(edges)
    # most bins should be exactly bin_width wide (continuum); only peak
    # regions are wider/narrower
    n_at_bin_width = int(np.sum(np.isclose(widths, 1.0, atol=1e-6)))
    assert n_at_bin_width > 50  # majority of the 100-unit range


def test_hist_uniform_with_peaks_preserves_total_counts():
    data = _peaked_data()
    out = hist_uniform_with_peaks(data, bin_width=1.0, prebin_width=0.05, n_sigma=5.0)
    assert out.values().sum() == pytest.approx(len(data))


def test_hist_uniform_with_peaks_with_range():
    data = _peaked_data()
    out = hist_uniform_with_peaks(
        data,
        bin_width=1.0,
        prebin_width=0.05,
        prebin_low=10.0,
        prebin_high=90.0,
        n_sigma=5.0,
    )
    edges = out.axes[0].edges
    assert edges[0] == 10.0
    in_range = int(np.sum((data >= 10.0) & (data < 90.0)))
    assert out.values().sum() == pytest.approx(in_range)


def test_rebin_uniform_with_peaks_preserves_total_counts():
    data = _peaked_data()
    fine = bh.Hist(bh.axis.Regular(2000, 0.0, 100.0), storage=bh.storage.Weight())
    fine.fill(data)
    out = rebin_uniform_with_peaks(fine, bin_width=1.0, n_sigma=5.0)
    assert out.values().sum() == pytest.approx(fine.values().sum())


def test_rebin_uniform_with_peaks_edges_are_subset_of_fine_grid():
    data = _peaked_data()
    fine = bh.Hist(bh.axis.Regular(2000, 0.0, 100.0), storage=bh.storage.Weight())
    fine.fill(data)
    out = rebin_uniform_with_peaks(fine, bin_width=1.0, n_sigma=5.0)
    assert np.all(np.isin(out.axes[0].edges, fine.axes[0].edges))


def test_rebin_uniform_with_peaks_continuum_near_bin_width():
    data = _peaked_data()
    fine = bh.Hist(bh.axis.Regular(2000, 0.0, 100.0), storage=bh.storage.Weight())
    fine.fill(data)
    out = rebin_uniform_with_peaks(fine, bin_width=1.0, n_sigma=5.0)
    widths = np.diff(out.axes[0].edges)
    n_continuum = int(np.sum(np.isclose(widths, 1.0, atol=0.05)))
    assert n_continuum > 50


def test_hist_bblocks_with_peaks_with_range():
    data = _peaked_data()
    out = hist_bblocks_with_peaks(
        data,
        prebin_width=0.05,
        prebin_low=10.0,
        prebin_high=90.0,
        n_sigma=5.0,
    )
    edges = out.axes[0].edges
    assert edges[0] == 10.0
    in_range = int(np.sum((data >= 10.0) & (data < 90.0)))
    assert out.values().sum() == pytest.approx(in_range)


# ---------------------------------------------------------------------------
# zero-peaks paths through all four *_with_peaks public entry points
# ---------------------------------------------------------------------------


def _flat_data(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 10.0, 10_000)


def test_hist_bblocks_with_peaks_zero_peaks_returns_variable_axis():
    data = _flat_data()
    out = hist_bblocks_with_peaks(data, prebin_width=0.1, n_sigma=5.0)
    assert isinstance(out.axes[0], bh.axis.Variable)
    assert out.values().sum() == pytest.approx(len(data))


def test_rebin_bblocks_with_peaks_zero_peaks_returns_variable_axis():
    data = _flat_data()
    fine = bh.Hist(bh.axis.Regular(100, 0.0, 10.0), storage=bh.storage.Weight())
    fine.fill(data)
    out = rebin_bblocks_with_peaks(fine, n_sigma=5.0)
    # _collapse_peaks must promote to Variable even when no peaks were found
    assert isinstance(out.axes[0], bh.axis.Variable)
    assert out.values().sum() == pytest.approx(fine.values().sum())


def test_hist_uniform_with_peaks_zero_peaks_returns_uniform_grid():
    data = _flat_data()
    out = hist_uniform_with_peaks(data, bin_width=1.0, prebin_width=0.1, n_sigma=5.0)
    widths = np.diff(out.axes[0].edges)
    # all bins should be close to bin_width when no peaks were inserted
    assert np.allclose(widths, 1.0, atol=0.01)


def test_rebin_uniform_with_peaks_zero_peaks_returns_uniform_grid():
    data = _flat_data()
    fine = bh.Hist(bh.axis.Regular(1000, 0.0, 10.0), storage=bh.storage.Weight())
    fine.fill(data)
    out = rebin_uniform_with_peaks(fine, bin_width=1.0, n_sigma=5.0)
    widths = np.diff(out.axes[0].edges)
    # snapped to fine grid (0.01 wide), so bins should be within ~0.01 of 1.0
    assert np.allclose(widths, 1.0, atol=0.02)


# ---------------------------------------------------------------------------
# _snap_indices
# ---------------------------------------------------------------------------


def test_snap_indices_exact_targets_return_exact_indices():
    from pygama.math.rebin import _snap_indices

    fine = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    idx = _snap_indices(fine, np.array([0.0, 2.0, 4.0]))
    assert np.array_equal(idx, [0, 2, 4])


def test_snap_indices_misaligned_targets_snap_to_nearest():
    from pygama.math.rebin import _snap_indices

    fine = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    # 1.4 → nearest 1, 2.6 → nearest 3
    idx = _snap_indices(fine, np.array([0.0, 1.4, 2.6, 4.0]))
    assert np.array_equal(idx, [0, 1, 3, 4])


def test_snap_indices_endpoints_clamped_and_duplicates_removed():
    from pygama.math.rebin import _snap_indices

    fine = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    # two close targets snap to the same fine index — dedup must collapse
    idx = _snap_indices(fine, np.array([0.0, 1.1, 1.2, 4.0]))
    # both 1.1 and 1.2 → 1; result must be strictly increasing
    assert np.all(np.diff(idx) > 0)
    # endpoints must be 0 and len(fine) - 1
    assert idx[0] == 0
    assert idx[-1] == len(fine) - 1
