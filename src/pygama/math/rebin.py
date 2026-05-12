"""Adaptive rebinning routines for 1D histograms.

Provides:

- :func:`hist_bblocks` and :func:`rebin_bblocks`: Bayesian-blocks
  adaptive binning (Scargle et al. 2013), backed by
  :func:`hepstats.modeling.bayesian_blocks`.
- :func:`collapse_peaks`: merges bins on each side of each detected
  peak by walking outward while the rate is descending and bin widths
  are not yet jumping up to continuum scale. Designed for histograms
  with adaptive bin widths (e.g. the output of :func:`rebin_bblocks`).

All routines return a ``Variable``-axis ``Weight``-storage
:class:`hist.Hist`.
"""

from __future__ import annotations

import awkward as ak
import hist as bh
import numpy as np
from hepstats.modeling import bayesian_blocks
from scipy import signal


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
    if isinstance(data, ak.Array):
        data = np.asarray(ak.ravel(data))

    if prebin_width is not None:
        if not (np.isfinite(prebin_width) and prebin_width > 0):
            msg = f"prebin_width must be finite and > 0, got {prebin_width}"
            raise ValueError(msg)
        lo = float(np.min(data)) if prebin_low is None else float(prebin_low)
        hi = float(np.max(data)) if prebin_high is None else float(prebin_high)
        if hi <= lo:
            msg = f"prebin_high ({hi}) must be greater than prebin_low ({lo})"
            raise ValueError(msg)
        nbins = max(1, int(np.ceil((hi - lo) / prebin_width)))
        # ensure the requested upper edge falls strictly inside the last bin
        # (boost_histogram's Regular axis is right-exclusive)
        if lo + nbins * prebin_width <= hi:
            nbins += 1
        fine = bh.Hist(
            bh.axis.Regular(nbins, lo, lo + nbins * prebin_width),
            storage=bh.storage.Weight(),
        )
        # mask explicitly to [lo, hi): the regular axis extends slightly past
        # hi to satisfy the prebin_width constraint, so unmasked data in
        # [hi, last_edge) would otherwise be silently included.
        if prebin_low is not None or prebin_high is not None:
            data = data[(data >= lo) & (data < hi)]
        fine.fill(data)
        return rebin_bblocks(fine, p0=p0)

    if prebin_low is not None or prebin_high is not None:
        msg = "prebin_low/prebin_high are only valid when prebin_width is given"
        raise ValueError(msg)

    edges = bayesian_blocks(data, p0=p0).astype(float)
    # boost_histogram's Variable axis is right-exclusive; bayesian_blocks
    # places the last edge at data.max(), so nudge it up by one ULP to
    # include the rightmost data point in the last bin.
    edges[-1] = np.nextafter(edges[-1], np.inf)
    h = bh.Hist(bh.axis.Variable(edges), storage=bh.storage.Weight())
    h.fill(data)
    return h


def rebin_bblocks(
    h: bh.Hist,
    *,
    p0: float = 0.05,
) -> bh.Hist:
    """Rebin a 1D :class:`hist.Hist` using Bayesian-blocks adaptive binning.

    Uses :func:`hepstats.modeling.bayesian_blocks` treating each
    fine-bin center as a weighted event (weight = bin count), snaps
    the resulting block edges to the fine-bin grid, and aggregates the
    input histogram onto the new edges by summing counts and variances.

    Parameters
    ----------
    h
        Input finely-binned 1D histogram. If the storage does not
        track variances, Poisson statistics are assumed.
    p0
        False-alarm probability for change-point detection.

    Returns
    -------
    out
        A ``Variable``-axis ``Weight``-storage histogram whose edges
        are a subset of the input edges.
    """
    if len(h.axes) != 1:
        msg = f"rebin_bblocks only supports 1D histograms, got {len(h.axes)}D"
        raise ValueError(msg)

    fine_edges = h.axes[0].edges
    centers = h.axes[0].centers
    values = h.values()
    variances = h.variances()
    if variances is None:
        variances = values  # Poisson statistics: variance = counts

    # hepstats raises RuntimeWarning on log(0) for empty bins; strip them
    # before running the algorithm — the snap step below maps block edges
    # back to fine_edges regardless, so no information is lost.
    nonzero = values > 0
    block_edges = bayesian_blocks(centers[nonzero], weights=values[nonzero], p0=p0)

    # snap block edges to the nearest fine-bin edge
    right = np.searchsorted(fine_edges, block_edges)
    left = np.clip(right - 1, 0, len(fine_edges) - 1)
    right = np.clip(right, 0, len(fine_edges) - 1)
    left_dist = np.abs(block_edges - fine_edges[left])
    right_dist = np.abs(block_edges - fine_edges[right])
    snap_idx = np.where(left_dist <= right_dist, left, right)
    # endpoints must coincide with the input range
    snap_idx[0] = 0
    snap_idx[-1] = len(fine_edges) - 1
    snap_idx = np.unique(snap_idx)

    new_edges = fine_edges[snap_idx]
    new_values = np.add.reduceat(values, snap_idx[:-1])
    new_variances = np.add.reduceat(variances, snap_idx[:-1])

    out = bh.Hist(bh.axis.Variable(new_edges), storage=bh.storage.Weight())
    view = np.asarray(out.view())
    view["value"] = new_values
    view["variance"] = new_variances
    return out


def collapse_peaks(
    h: bh.Hist,
    *,
    n_sigma: float = 5.0,
    width_factor: float = 5.0,
    max_bin_width: float | None = None,
) -> bh.Hist:
    """Merge bins on each side of detected peaks using the peak shape.

    Intended for histograms with adaptive bin widths (e.g. the output
    of :func:`rebin_bblocks`) and **non-negative bin contents**.

    Peaks are detected as strict local maxima of the rate
    (counts/width) — the right signal for variable-width histograms,
    where a real peak may have fewer counts than a wider neighbor but
    still a higher rate. Candidates are filtered by significance
    computed from their prominence (see
    :func:`scipy.signal.peak_prominences`):

        significance = prominence * w_peak / sqrt(n_peak)

    where ``w_peak`` and ``n_peak`` are the width and counts of the
    peak bin alone (*not* the eventual merged region). The threshold
    is ``n_sigma``. For peaks that bblocks resolves into several
    adjacent narrow bins this underestimates the true significance.

    For each surviving peak the walk extends outward on both sides
    while all of the following hold:

    * the rate is strictly decreasing — still on the descending side
      of the peak;
    * the next bin's width is below ``min(width_factor * peak_width,
      max_bin_width)`` — the walk has not grown beyond the natural
      scale of the peak;
    * the next bin is not a strict local minimum of rate — close
      peaks aren't merged through a shared valley.

    Note that :func:`scipy.signal.find_peaks` never reports the
    boundary bins (indices 0 or ``n - 1``), so a spike at the first
    or last bin is silently passed through.

    Parameters
    ----------
    h
        Input 1D histogram with (typically) variable-width bins. Bin
        contents must be non-negative. Poisson statistics are assumed
        if the storage does not track variances.
    n_sigma
        Significance threshold for peak detection. Must be > 0.
    width_factor
        Peak-relative cap on the walk width: the walk stops when a
        neighboring bin is ``width_factor`` times wider than the peak
        bin itself. Encodes the resolution-driven width of real peaks.
        Must be > 1 so the peak bin's immediate neighbors are eligible
        when no wider than the peak itself.
    max_bin_width
        Optional absolute cap (in axis units), applied via
        ``min(width_factor * peak_width, max_bin_width)``. ``None``
        disables it.

    Returns
    -------
    out
        A ``Variable``-axis ``Weight``-storage histogram whose edges
        are a subset of the input edges.
    """
    if len(h.axes) != 1:
        msg = f"collapse_peaks only supports 1D histograms, got {len(h.axes)}D"
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
        msg = "collapse_peaks requires non-negative bin contents"
        raise ValueError(msg)
    variances = h.variances()
    if variances is None:
        variances = values  # Poisson statistics: variance = counts

    # detect peaks as strict local maxima of the rate (counts/width):
    # the right signal for variable-width histograms, where a real peak
    # can have fewer counts than a wider neighbor but a higher rate.
    rate = values / widths
    peaks, _ = signal.find_peaks(rate)

    if len(peaks) > 0:
        # significance from scipy's prominence (the rate drop from the
        # peak down to its lowest surrounding contour that doesn't reach
        # a higher peak — i.e. the local baseline). For a peak with
        # rate r_p in a bin of width w_p and a baseline rate r_b, the
        # excess counts at the peak above what the baseline would
        # predict is (r_p - r_b) * w_p = prominence_rate * w_p; the
        # Poisson sigma of the peak counts is sqrt(n_p). Their ratio
        # is the prominence-based significance.
        prominences, _, _ = signal.peak_prominences(rate, peaks)
        excess_counts = prominences * widths[peaks]
        sigma = np.sqrt(np.maximum(values[peaks], 1.0))
        significance = excess_counts / sigma
        peaks = peaks[significance >= n_sigma]

    if len(peaks) == 0:
        out = bh.Hist(bh.axis.Variable(fine_edges), storage=bh.storage.Weight())
        view = np.asarray(out.view())
        view["value"] = values
        view["variance"] = variances
        return out

    # walk outward from each peak while rate descends and bin widths stay
    # within `width_factor` times the peak bin's own width. Additionally
    # stop before stepping into a local minimum of rate (the valley
    # between adjacent peaks), so close peaks don't share a tail bin.
    n = len(values)
    cap = max_bin_width if max_bin_width is not None else np.inf
    regions: list[tuple[int, int]] = []
    for p in peaks:
        width_cap = min(widths[p] * width_factor, cap)
        li = _walk_peak(p, -1, rate, widths, width_cap, n)
        ri = _walk_peak(p, +1, rate, widths, width_cap, n)
        regions.append((li, ri))

    # resolve overlaps: sweep left-to-right and merge regions that actually
    # share at least one bin. regions that are merely adjacent (one peak's
    # walk stopped at bin k and another's started at bin k+1) stay separate
    # so the boundary edge between them is preserved in the output.
    regions.sort()
    merged: list[list[int]] = []
    for li, ri in regions:
        if merged and li <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], ri)
        else:
            merged.append([li, ri])

    # drop interior edges of each merged region
    keep = np.ones(len(fine_edges), dtype=bool)
    for li, ri in merged:
        keep[li + 1 : ri + 1] = False
    snap_idx = np.where(keep)[0]

    new_edges = fine_edges[snap_idx]
    new_values = np.add.reduceat(values, snap_idx[:-1])
    new_variances = np.add.reduceat(variances, snap_idx[:-1])

    out = bh.Hist(bh.axis.Variable(new_edges), storage=bh.storage.Weight())
    view = np.asarray(out.view())
    view["value"] = new_values
    view["variance"] = new_variances
    return out
