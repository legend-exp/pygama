"""Adaptive rebinning routines for 1D histograms.

Two adaptive strategies are offered:

- **Bayesian blocks** (Scargle et al. 2013), backed by
  :func:`hepstats.modeling.bayesian_blocks`.
- **Peak-aware**: peaks are detected and each peak region is collapsed
  into a single bin. The continuum is either left at bblocks spacing or
  rebinned to a uniform width.

Functions taking a raw data array are named ``hist_*``; those taking a
finely-binned :class:`hist.Hist` are named ``rebin_*``. All routines
return a ``Variable``-axis ``Weight``-storage :class:`hist.Hist`.
"""

from __future__ import annotations

import awkward as ak
import hist as bh
import numpy as np
from hepstats.modeling import bayesian_blocks
from scipy import signal

# ---------------------------------------------------------------------------
# private primitives
# ---------------------------------------------------------------------------


def _prepare_data(
    data: np.ndarray | ak.Array,
    prebin_low: float | None,
    prebin_high: float | None,
) -> tuple[np.ndarray, float, float]:
    """Flatten awkward inputs, derive the data range, and apply the
    ``[prebin_low, prebin_high)`` mask when a range is requested.

    Returns ``(data, lo, hi)`` with ``lo`` and ``hi`` set to the user-
    supplied bounds or to ``data.min()``/``data.max()`` as default.
    """
    if isinstance(data, ak.Array):
        data = np.asarray(ak.ravel(data))
    if prebin_low is None and prebin_high is None:
        return data, float(np.min(data)), float(np.max(data))
    lo = float(prebin_low) if prebin_low is not None else float(np.min(data))
    hi = float(prebin_high) if prebin_high is not None else float(np.max(data))
    if hi <= lo:
        msg = f"prebin_high ({hi}) must be greater than prebin_low ({lo})"
        raise ValueError(msg)
    return data[(data >= lo) & (data < hi)], lo, hi


def _walk_peak(
    start: int,
    step: int,
    rate: np.ndarray,
    widths: np.ndarray,
    width_cap: float,
    n: int,
) -> int:
    """Walk outward from a peak bin and return the farthest index reached.

    Steps in direction ``step`` (-1 or +1), stopping when the rate
    stops descending, the next bin's width reaches ``width_cap``, or
    the next bin is a strict local minimum of rate. At the array
    boundary the missing neighbor is treated as +inf.
    """
    i = start
    while 0 <= i + step <= n - 1:
        j = i + step
        if rate[j] >= rate[i] or widths[j] >= width_cap:
            break
        next_j = j + step
        next_rate = rate[next_j] if 0 <= next_j <= n - 1 else np.inf
        if next_rate >= rate[j]:
            break
        i = j
    return i


def _bblocks_edges(
    data: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    p0: float = 0.05,
) -> np.ndarray:
    """Bayesian-blocks edges from raw data; last edge nudged by one ULP
    so :class:`bh.axis.Variable` (right-exclusive) includes ``data.max()``.
    """
    edges = bayesian_blocks(data, weights=weights, p0=p0).astype(float)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    return edges


def _snap_indices(fine_edges: np.ndarray, target_edges: np.ndarray) -> np.ndarray:
    """Nearest-edge indices into ``fine_edges`` for each ``target_edges``.

    Endpoints are clamped to ``0`` and ``len(fine_edges) - 1`` and
    duplicates are removed, so the result is a strictly increasing
    integer array suitable as group boundaries for
    :func:`hist.rebin`.
    """
    right = np.searchsorted(fine_edges, target_edges)
    left = np.clip(right - 1, 0, len(fine_edges) - 1)
    right = np.clip(right, 0, len(fine_edges) - 1)
    left_dist = np.abs(target_edges - fine_edges[left])
    right_dist = np.abs(target_edges - fine_edges[right])
    idx = np.where(left_dist <= right_dist, left, right)
    idx[0] = 0
    idx[-1] = len(fine_edges) - 1
    return np.unique(idx)


def _bblocks_edges_from_hist(h: bh.Hist, *, p0: float = 0.05) -> np.ndarray:
    """Bayesian-blocks edges from a finely-binned histogram, snapped to
    ``h.axes[0].edges``. Empty bins are dropped to avoid the ``log(0)``
    warning hepstats raises on zero-weight events.
    """
    if len(h.axes) != 1:
        msg = f"only 1D histograms supported, got {len(h.axes)}D"
        raise ValueError(msg)

    fine_edges = h.axes[0].edges
    centers = h.axes[0].centers
    values = h.values()
    nonzero = values > 0
    block_edges = bayesian_blocks(centers[nonzero], weights=values[nonzero], p0=p0)
    return fine_edges[_snap_indices(fine_edges, block_edges)]


def _find_peak_regions(
    h: bh.Hist,
    *,
    n_sigma: float = 5.0,
    width_factor: float = 5.0,
    max_bin_width: float | None = None,
) -> np.ndarray:
    """Detect peaks in an adaptive-bin histogram and return the merged
    region edges in axis units. Shape ``(n_regions, 2)``; empty array
    if no peaks are detected.

    Peaks are strict local maxima of the rate (counts/width), filtered
    by significance ``prominence * w_peak / sqrt(n_peak)`` against
    ``n_sigma``. Each surviving peak is grown outward via
    :func:`_walk_peak` and overlapping regions are merged.
    """
    if len(h.axes) != 1:
        msg = f"only 1D histograms supported, got {len(h.axes)}D"
        raise ValueError(msg)
    if not (np.isfinite(n_sigma) and n_sigma > 0):
        msg = f"n_sigma must be finite and > 0, got {n_sigma}"
        raise ValueError(msg)
    if not (np.isfinite(width_factor) and width_factor > 1):
        msg = f"width_factor must be finite and > 1, got {width_factor}"
        raise ValueError(msg)
    if max_bin_width is not None and not (
        np.isfinite(max_bin_width) and max_bin_width > 0
    ):
        msg = f"max_bin_width must be finite and > 0, got {max_bin_width}"
        raise ValueError(msg)

    fine_edges = h.axes[0].edges
    widths = np.diff(fine_edges)
    if np.any(widths <= 0):
        msg = "histogram has bins with zero or negative width"
        raise ValueError(msg)
    values = h.values()
    if np.any(values < 0):
        msg = "peak detection requires non-negative bin contents"
        raise ValueError(msg)

    rate = values / widths
    peaks, _ = signal.find_peaks(rate)
    if len(peaks) == 0:
        return np.empty((0, 2), dtype=float)

    prominences, _, _ = signal.peak_prominences(rate, peaks)
    significance = prominences * widths[peaks] / np.sqrt(np.maximum(values[peaks], 1.0))
    peaks = peaks[significance >= n_sigma]
    if len(peaks) == 0:
        return np.empty((0, 2), dtype=float)

    n = len(values)
    cap = max_bin_width if max_bin_width is not None else np.inf
    regions: list[tuple[int, int]] = []
    for p in peaks:
        width_cap = min(widths[p] * width_factor, cap)
        li = _walk_peak(p, -1, rate, widths, width_cap, n)
        ri = _walk_peak(p, +1, rate, widths, width_cap, n)
        regions.append((li, ri))

    # merge overlapping regions; adjacent (touching) regions stay separate
    regions.sort()
    merged: list[list[int]] = []
    for li, ri in regions:
        if merged and li <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], ri)
        else:
            merged.append([li, ri])

    return np.array([[fine_edges[li], fine_edges[ri + 1]] for li, ri in merged])


def _collapse_peaks(
    h: bh.Hist,
    *,
    n_sigma: float = 5.0,
    width_factor: float = 5.0,
    max_bin_width: float | None = None,
) -> bh.Hist:
    """Merge bins inside each detected peak region into a single bin.

    Output edges are a subset of the input edges.
    """
    regions = _find_peak_regions(
        h,
        n_sigma=n_sigma,
        width_factor=width_factor,
        max_bin_width=max_bin_width,
    )
    fine_edges = h.axes[0].edges

    keep = np.ones(len(fine_edges), dtype=bool)
    for lo_p, hi_p in regions:
        li = int(np.searchsorted(fine_edges, lo_p))
        ri = int(np.searchsorted(fine_edges, hi_p))
        keep[li + 1 : ri] = False
    # always route through bh.rebin so the output axis is uniformly Variable,
    # even when no consolidation happens (a plain copy would preserve a
    # Regular axis input, violating the module contract).
    return h[:: bh.rebin(edges=fine_edges[keep].tolist())]


def _uniform_edges_with_peaks(
    low: float,
    high: float,
    *,
    bin_width: float,
    peak_regions: np.ndarray,
) -> np.ndarray:
    """Uniform grid in ``[low, high]`` with each peak region inserted
    as a single bin, replacing any uniform edges strictly inside it.

    Assumes ``peak_regions`` is sorted and non-overlapping (the output
    of :func:`_find_peak_regions` is). Peak regions outside
    ``[low, high]`` produce out-of-range edges in the result; the
    caller is responsible for ensuring the range covers them.
    """
    if not (np.isfinite(bin_width) and bin_width > 0):
        msg = f"bin_width must be finite and > 0, got {bin_width}"
        raise ValueError(msg)
    if high <= low:
        msg = f"high ({high}) must be greater than low ({low})"
        raise ValueError(msg)

    n_bins = max(1, int(np.ceil((high - low) / bin_width)))
    uniform = low + np.arange(n_bins + 1) * bin_width
    if uniform[-1] > high:
        uniform[-1] = high
    elif uniform[-1] < high:
        uniform = np.append(uniform, high)

    if peak_regions.shape[0] == 0:
        return uniform

    keep_mask = np.ones(len(uniform), dtype=bool)
    for lo_p, hi_p in peak_regions:
        keep_mask &= ~((uniform > lo_p) & (uniform < hi_p))
    return np.unique(np.concatenate([uniform[keep_mask], peak_regions.flatten()]))


# ---------------------------------------------------------------------------
# public API — raw data → histogram
# ---------------------------------------------------------------------------


def hist_bblocks(
    data: np.ndarray | ak.Array,
    *,
    prebin_width: float | None = None,
    prebin_low: float | None = None,
    prebin_high: float | None = None,
    p0: float = 0.05,
) -> bh.Hist:
    """Histogram an unbinned data array using Bayesian-blocks edges.

    By default runs :func:`hepstats.modeling.bayesian_blocks` directly
    on ``data``, which is O(N²) in sample size. Passing
    ``prebin_width`` first bins ``data`` into a uniform fine histogram
    and calls :func:`rebin_bblocks` instead, which is the faster path
    for large samples.

    Awkward inputs are flattened across all axes before binning.

    Parameters
    ----------
    data
        Array of measurement values.
    prebin_width
        Pre-binning width. Must be much finer than the smallest
        feature to resolve; resolution loss propagates to the output.
    prebin_low, prebin_high
        Range of the pre-binned histogram. Default to ``data.min()``
        / ``data.max()``. Data outside ``[prebin_low, prebin_high)``
        is discarded. Only valid when ``prebin_width`` is given. The
        effective upper edge may exceed ``prebin_high`` by up to
        ``prebin_width`` to cover the range with an integer number
        of bins.
    p0
        False-alarm probability for change-point detection.

    Returns
    -------
    h
        A ``Variable``-axis ``Weight``-storage histogram filled with
        ``data``.
    """
    if prebin_width is None:
        if prebin_low is not None or prebin_high is not None:
            msg = "prebin_low/prebin_high are only valid when prebin_width is given"
            raise ValueError(msg)
        if isinstance(data, ak.Array):
            data = np.asarray(ak.ravel(data))
        edges = _bblocks_edges(data, p0=p0)
        h = bh.Hist(bh.axis.Variable(edges), storage=bh.storage.Weight())
        h.fill(data)
        return h

    if not (np.isfinite(prebin_width) and prebin_width > 0):
        msg = f"prebin_width must be finite and > 0, got {prebin_width}"
        raise ValueError(msg)
    data, lo, hi = _prepare_data(data, prebin_low, prebin_high)
    # right-exclusive Regular axis: extend by one bin if needed so hi is
    # strictly inside the last bin
    nbins = max(1, int(np.ceil((hi - lo) / prebin_width)))
    if lo + nbins * prebin_width <= hi:
        nbins += 1
    fine = bh.Hist(
        bh.axis.Regular(nbins, lo, lo + nbins * prebin_width),
        storage=bh.storage.Weight(),
    )
    fine.fill(data)
    return rebin_bblocks(fine, p0=p0)


def hist_bblocks_with_peaks(
    data: np.ndarray | ak.Array,
    *,
    prebin_width: float | None = None,
    prebin_low: float | None = None,
    prebin_high: float | None = None,
    p0: float = 0.05,
    n_sigma: float = 5.0,
    width_factor: float = 5.0,
    max_bin_width: float | None = None,
) -> bh.Hist:
    """Histogram raw data with Bayesian-blocks edges, then collapse
    each detected peak region into a single bin.

    Equivalent to applying peak consolidation to the output of
    :func:`hist_bblocks`. See those two functions for the meaning of
    each parameter.
    """
    bb = hist_bblocks(
        data,
        prebin_width=prebin_width,
        prebin_low=prebin_low,
        prebin_high=prebin_high,
        p0=p0,
    )
    return _collapse_peaks(
        bb,
        n_sigma=n_sigma,
        width_factor=width_factor,
        max_bin_width=max_bin_width,
    )


def hist_uniform_with_peaks(
    data: np.ndarray | ak.Array,
    *,
    bin_width: float,
    prebin_width: float | None = None,
    prebin_low: float | None = None,
    prebin_high: float | None = None,
    p0: float = 0.05,
    n_sigma: float = 5.0,
    width_factor: float = 5.0,
    max_bin_width: float | None = None,
) -> bh.Hist:
    """Histogram raw data with uniform-width continuum bins, with each
    detected peak collapsed to a single bin.

    Parameters
    ----------
    bin_width
        Uniform continuum bin width. The output spans
        ``[prebin_low, prebin_high)`` (or ``[data.min(), data.max()]``
        if not given); bins overlapping any detected peak region are
        replaced by a single peak-region bin.
    prebin_width, p0
        Forwarded to the internal :func:`hist_bblocks` peak-detection
        pass.
    prebin_low, prebin_high
        Define both the output range and (after masking ``data`` to
        ``[prebin_low, prebin_high)``) the input range of the
        peak-detection pass.
    n_sigma, width_factor, max_bin_width
        Forwarded to peak detection; see :func:`_find_peak_regions`.

    Returns
    -------
    h
        A ``Variable``-axis ``Weight``-storage histogram filled with
        ``data``.
    """
    data, lo, hi = _prepare_data(data, prebin_low, prebin_high)
    bb = hist_bblocks(data, prebin_width=prebin_width, p0=p0)
    regions = _find_peak_regions(
        bb,
        n_sigma=n_sigma,
        width_factor=width_factor,
        max_bin_width=max_bin_width,
    )

    new_edges = _uniform_edges_with_peaks(
        lo, hi, bin_width=bin_width, peak_regions=regions
    ).copy()
    new_edges[-1] = np.nextafter(new_edges[-1], np.inf)
    out = bh.Hist(bh.axis.Variable(new_edges), storage=bh.storage.Weight())
    out.fill(data)
    return out


# ---------------------------------------------------------------------------
# public API — fine histogram → coarser histogram
# ---------------------------------------------------------------------------


def rebin_bblocks(h: bh.Hist, *, p0: float = 0.05) -> bh.Hist:
    """Rebin a 1D :class:`hist.Hist` using Bayesian-blocks adaptive binning.

    Uses :func:`hepstats.modeling.bayesian_blocks` treating each
    fine-bin center as a weighted event (weight = bin count), snaps
    the resulting block edges to the fine-bin grid, and aggregates the
    input histogram onto the new edges.

    Parameters
    ----------
    h
        Input finely-binned 1D histogram.
    p0
        False-alarm probability for change-point detection.

    Returns
    -------
    out
        A ``Variable``-axis ``Weight``-storage histogram whose edges
        are a subset of the input edges.
    """
    # _bblocks_edges_from_hist returns edges that are exact members of
    # h.axes[0].edges, so they pass bh.rebin's edge-membership check directly.
    return h[:: bh.rebin(edges=_bblocks_edges_from_hist(h, p0=p0).tolist())]


def rebin_bblocks_with_peaks(
    h: bh.Hist,
    *,
    n_sigma: float = 5.0,
    width_factor: float = 5.0,
    max_bin_width: float | None = None,
    p0: float = 0.05,
) -> bh.Hist:
    """Rebin a 1D :class:`hist.Hist` with Bayesian-blocks adaptive
    binning, then collapse each detected peak region into a single bin.

    See :func:`rebin_bblocks` and :func:`_find_peak_regions` for the
    meaning of each parameter.
    """
    bb = rebin_bblocks(h, p0=p0)
    return _collapse_peaks(
        bb,
        n_sigma=n_sigma,
        width_factor=width_factor,
        max_bin_width=max_bin_width,
    )


def rebin_uniform_with_peaks(
    h: bh.Hist,
    *,
    bin_width: float,
    n_sigma: float = 5.0,
    width_factor: float = 5.0,
    max_bin_width: float | None = None,
    p0: float = 0.05,
) -> bh.Hist:
    """Rebin a 1D :class:`hist.Hist` onto uniform continuum bins, with
    each detected peak region kept as a single bin.

    Peak detection runs on an internal :func:`rebin_bblocks` pass.
    Output edges are snapped to ``h.axes[0].edges``, so ``bin_width``
    need not align exactly with the input grid.

    Parameters
    ----------
    h
        Input finely-binned 1D histogram.
    bin_width
        Uniform continuum bin width (in axis units).
    n_sigma, width_factor, max_bin_width
        Forwarded to peak detection; see :func:`_find_peak_regions`.
    p0
        False-alarm probability for the internal bblocks pass.

    Returns
    -------
    out
        A ``Variable``-axis ``Weight``-storage histogram whose edges
        are a subset of the input edges.
    """
    bb = rebin_bblocks(h, p0=p0)
    regions = _find_peak_regions(
        bb,
        n_sigma=n_sigma,
        width_factor=width_factor,
        max_bin_width=max_bin_width,
    )
    fine_edges = h.axes[0].edges
    target_edges = _uniform_edges_with_peaks(
        float(fine_edges[0]),
        float(fine_edges[-1]),
        bin_width=bin_width,
        peak_regions=regions,
    )
    snap_idx = _snap_indices(fine_edges, target_edges)
    return h[:: bh.rebin(edges=fine_edges[snap_idx].tolist())]
