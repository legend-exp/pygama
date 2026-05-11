"""Bayesian-blocks adaptive rebinning for 1D histograms.

Provides routines for adaptive binning based on the Bayesian-blocks
algorithm (Scargle et al. 2013), backed by
:func:`astropy.stats.bayesian_blocks`.

Two entry points are provided:

- :func:`hist_bblocks` histograms an unbinned data array
  (:class:`numpy.ndarray` or :class:`awkward.Array`) using
  Bayesian-blocks edges.
- :func:`rebin_bblocks` rebins a finely-binned :class:`hist.Hist`
  using the ``measures`` fitness.

Both return a ``Variable``-axis ``Weight``-storage :class:`hist.Hist`.
"""

from __future__ import annotations

import awkward as ak
import hist as bh
import numpy as np
from astropy.stats import bayesian_blocks


def hist_bblocks(
    data: np.ndarray | ak.Array,
    *,
    prebin_width: float | None = None,
    prebin_low: float | None = None,
    prebin_high: float | None = None,
    p0: float = 0.05,
) -> bh.Hist:
    """Histogram an unbinned data array using Bayesian-blocks edges.

    By default uses :func:`astropy.stats.bayesian_blocks` with
    ``fitness='events'`` directly on ``data``. The ``events`` fitness
    is O(N²) in the number of points, which becomes prohibitive
    for large samples; passing ``prebin_width`` first bins the data into
    a uniform fine histogram of that width and then runs
    :func:`rebin_bblocks` (``measures`` fitness) on it, which is
    O(Nₙ²) in the number of fine bins.

    Awkward inputs are flattened across all axes before binning.

    Parameters
    ----------
    data
        Array of measurement values.
    prebin_width
        Optional pre-binning width. If given, ``data`` is first
        histogrammed into uniform fine bins of this width and the
        result is rebinned with :func:`rebin_bblocks`. Choose a
        ``prebin_width`` much finer than the smallest feature you expect
        to resolve; otherwise the resolution loss propagates to the
        output edges.
    prebin_low, prebin_high
        Optional lower / upper edge of the pre-binned histogram.
        Default to ``data.min()`` / ``data.max()``. Values outside
        ``[prebin_low, prebin_high)`` land in the overflow bins of
        the fine histogram and are excluded from the output. Only
        valid when ``prebin_width`` is given. The effective upper
        edge may exceed ``prebin_high`` by up to ``prebin_width`` so
        that an integer number of bins covers the requested range.
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
        # Filter data to [prebin_low, prebin_high) range when explicitly specified
        if prebin_low is not None or prebin_high is not None:
            mask = (data >= lo) & (data < hi)
            fine.fill(data[mask])
        else:
            fine.fill(data)
        return rebin_bblocks(fine, p0=p0)

    if prebin_low is not None or prebin_high is not None:
        msg = "prebin_low/prebin_high are only valid when prebin_width is given"
        raise ValueError(msg)

    edges = bayesian_blocks(data, fitness="events", p0=p0).astype(float)
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

    Uses :func:`astropy.stats.bayesian_blocks` with ``fitness='measures'``
    on the fine-bin centers, snaps the resulting block edges to the
    fine-bin grid, and aggregates the input histogram onto the new
    edges by summing counts and variances.

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

    sigma = np.sqrt(np.where(variances > 0, variances, 1.0))

    block_edges = bayesian_blocks(
        centers, x=values, sigma=sigma, fitness="measures", p0=p0
    )

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
