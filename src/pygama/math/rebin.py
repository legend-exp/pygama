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

Public functions
----------------

- :func:`hist_bblocks`: histogram raw data using Bayesian-blocks edges.
- :func:`hist_bblocks_with_peaks`: histogram raw data with bblocks edges
  on the continuum and one merged bin per detected peak.
- :func:`hist_uniform_with_peaks`: histogram raw data with a uniform
  continuum binning and one merged bin per detected peak.
- :func:`rebin_bblocks`: rebin an existing fine histogram onto
  Bayesian-blocks edges.
- :func:`rebin_bblocks_with_peaks`: rebin an existing fine histogram
  onto bblocks edges and merge each detected peak into a single bin.
- :func:`rebin_uniform_with_peaks`: rebin an existing fine histogram
  onto a uniform continuum binning and merge each detected peak into
  a single bin.

Rebinning a histogram with the edges of another
-----------------------------------------------

To rebin a fine histogram ``h_fine`` onto the edges of another
histogram ``h_ref``, use :func:`hist.rebin` directly (the edges of
``h_ref`` must be a subset of those of ``h_fine``):

>>> import hist as bh
>>> h_out = h_fine[:: bh.rebin(edges=h_ref.axes[0].edges.tolist())]
"""

from __future__ import annotations

import awkward as ak
import hist as bh
import numpy as np
from hepstats.modeling import bayesian_blocks
from scipy import signal


def _prepare_data(
    data: np.ndarray | ak.Array,
    prebin_low: float | None,
    prebin_high: float | None,
) -> tuple[np.ndarray, float, float]:
    """Flatten awkward inputs and apply the optional range mask."""
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


def _bblocks_edges(data: np.ndarray, **kwargs) -> np.ndarray:
    """Bblocks edges; last edge nudged by 1 ULP for right-exclusive axes."""
    edges = bayesian_blocks(data, **kwargs).astype(float)
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
    """Bayesian-blocks edges from a fine histogram, snapped to its grid.

    Empty bins are dropped to avoid the ``log(0)`` warning hepstats
    raises on zero-weight events.
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
    """Detect peaks and return ``(n_regions, 2)`` merged region edges.

    Edges are in axis units; the array is empty if no peaks pass the
    significance threshold. Peaks are strict local maxima of the rate
    (counts/width), filtered by ``prominence * w_peak / sqrt(n_peak)``
    against ``n_sigma``. When ``max_bin_width`` is given, candidates
    sitting on a bin wider than the cap are also rejected (line-like
    features have resolution-bounded widths; wide bins flag Compton
    edges and smooth-continuum plateaus). Each surviving peak is grown
    outward via :func:`_walk_peak` and overlapping regions are merged.
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
    if max_bin_width is not None:
        peaks = peaks[widths[peaks] < max_bin_width]
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
    """Merge bins inside each detected peak region into a single bin."""
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
    """Uniform grid over ``[low, high]`` with peak regions as single bins.

    Any uniform edge strictly inside a peak region, OR within
    ``bin_width`` of one of its boundaries, is dropped: the peak
    region's neighboring continuum bins absorb the would-be fragment,
    making them up to ``2 * bin_width`` wide instead. Uniform edges
    exactly on a peak boundary are kept. Assumes ``peak_regions`` is
    sorted and non-overlapping.
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
        in_zone = (uniform > lo_p - bin_width) & (uniform < hi_p + bin_width)
        is_boundary = (uniform == lo_p) | (uniform == hi_p)
        keep_mask &= ~(in_zone & ~is_boundary)
    return np.unique(np.concatenate([uniform[keep_mask], peak_regions.flatten()]))


def hist_bblocks(
    data: np.ndarray | ak.Array,
    *,
    prebin_width: float | None = None,
    prebin_low: float | None = None,
    prebin_high: float | None = None,
    p0: float = 0.05,
) -> bh.Hist:
    """Histogram an unbinned data array using Bayesian-blocks edges.

    For large samples, pass ``prebin_width`` to first bin the data
    on a uniform fine grid and then call :func:`rebin_bblocks` --
    this is much faster than running bblocks on the raw data
    directly. Awkward inputs are flattened before binning.

    Parameters
    ----------
    data
        Array of measurement values (numpy or awkward).
    prebin_width
        Width of the uniform pre-binning grid. Should be a few
        times finer than the narrowest feature you want to
        resolve (e.g. ~0.5 keV for HPGe lines with ~3 keV FWHM).
    prebin_low, prebin_high
        Histogram range. Default to ``data.min()`` / ``data.max()``.
        Data outside the range is discarded. Only valid together
        with ``prebin_width``. The upper edge may overshoot
        ``prebin_high`` by up to ``prebin_width`` so the range is
        covered by an integer number of bins.
    p0
        Bblocks sensitivity: lower values give coarser bins, higher
        values finer. Typical 0.01-0.1.

    Returns
    -------
    h
        A ``Variable``-axis ``Weight``-storage histogram filled
        with ``data``.

    Examples
    --------
    >>> from pygama.math.rebin import hist_bblocks
    >>> h = hist_bblocks(energies, prebin_width=0.5, prebin_low=0, prebin_high=3000)
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
    """Histogram raw data with Bayesian-blocks edges and collapsed peaks.

    Equivalent to ``_collapse_peaks(hist_bblocks(data, ...))``.

    Parameters
    ----------
    data, prebin_width, prebin_low, prebin_high, p0
        See :func:`hist_bblocks`.
    n_sigma, width_factor, max_bin_width
        See :func:`rebin_bblocks_with_peaks`.

    Returns
    -------
    h
        A ``Variable``-axis ``Weight``-storage histogram.

    Examples
    --------
    Starting with a large data array, prebin it for performance and use a
    maximum peak width of 10.

    >>> from pygama.math.rebin import hist_bblocks_with_peaks
    >>> data = [...]
    >>> h = hist_bblocks_with_peaks(
    ...     data, prebin_width=0.5, prebin_low=0, prebin_high=3000, max_bin_width=10
    ... )
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
    """Histogram raw data with uniform continuum bins and collapsed peaks.

    Parameters
    ----------
    data, prebin_width, p0
        See :func:`hist_bblocks`.
    prebin_low, prebin_high
        See :func:`hist_bblocks`; here they also define the
        output range.
    bin_width
        See :func:`rebin_uniform_with_peaks`.
    n_sigma, width_factor, max_bin_width
        See :func:`rebin_bblocks_with_peaks`.

    Returns
    -------
    h
        A ``Variable``-axis ``Weight``-storage histogram filled
        with ``data``.

    Examples
    --------
    Starting with a large data array, prebin it for performance and use a
    maximum peak width of 10. In between the peaks use an uniform bin width of
    2.

    >>> from pygama.math.rebin import hist_uniform_with_peaks
    >>> data = [...]
    >>> h = hist_uniform_with_peaks(
    ...     energies,
    ...     bin_width=2,
    ...     prebin_width=0.5,
    ...     prebin_low=0,
    ...     prebin_high=3000,
    ...     max_bin_width=10,
    ... )
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


def rebin_bblocks(h: bh.Hist, *, p0: float = 0.05) -> bh.Hist:
    """Rebin a fine 1D histogram using Bayesian-blocks adaptive binning.

    Each fine bin is treated as a weighted event (weight = bin count);
    the resulting block edges are snapped to the input grid.

    Parameters
    ----------
    h
        Input finely-binned 1D histogram.
    p0
        See :func:`hist_bblocks`.

    Returns
    -------
    out
        A ``Variable``-axis ``Weight``-storage histogram whose edges
        are a subset of the input edges.

    Examples
    --------
    >>> from pygama.math.rebin import rebin_bblocks
    >>> import hist
    >>> h_fine = hist.new.Reg(1000, 0, 1000).Double().fill(...)
    >>> h_bb = rebin_bblocks(h_fine)
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
    """Rebin a fine histogram with bblocks edges and collapsed peaks.

    First applies :func:`rebin_bblocks` to get adaptive continuum
    bins, then merges each detected gamma peak into a single bin.

    Parameters
    ----------
    h
        Input finely-binned 1D histogram.
    n_sigma
        How tall a peak must be above its local baseline to be
        flagged, in units of Poisson noise. Default ``5.0``.
        Lower values pick up weaker peaks but also more spurious
        detections.
    width_factor
        How far the merge can extend on each side of a peak,
        relative to the width of the peak's own bblocks bin.
        The walk stops at the first bin wider than
        ``width_factor * peak_bin_width``. Default ``5.0``;
        try ``2.0-3.0`` for tighter merges on HPGe spectra.
    max_bin_width
        Maximum allowed width for a peak bin (in axis units).
        Two effects:

        1. Detection: candidates whose bblocks bin is wider than
           this are **rejected** -- useful for filtering out
           Compton edges, continuum plateaus, and other broad
           features that the peak finder cannot tell apart from
           real gamma lines based on shape alone (see Notes).
        2. Walk: the merge walk never extends into a bin wider
           than this, regardless of ``width_factor``.

        Default ``None`` (no filter). **For HPGe spectra set it
        to a few times FWHM at the highest energy of interest
        (5-20 keV is typical)** to suppress non-line features.
    p0
        See :func:`hist_bblocks`.

    Returns
    -------
    out
        A ``Variable``-axis ``Weight``-storage histogram whose
        edges are a subset of the input edges.

    Notes
    -----
    The peak finder only looks at local maxima of the rate
    (counts/width) and how high they stand above their local
    baseline; it knows nothing about what a real gamma line
    looks like. It can therefore flag features that aren't
    line-like:

    * resolution-limited gamma peaks (the intended target);
    * the upper end of a long Compton-edge ramp;
    * step transitions;
    * wide continuum plateaus that bblocks captured in a single
      wide bin.

    Real HPGe peaks live on narrow bblocks bins (FWHM is a few
    keV even at multi-MeV energies); the other features end up
    on much wider bins. ``max_bin_width`` is the simplest filter
    to reject everything that isn't a real line.

    Examples
    --------
    >>> from pygama.math.rebin import rebin_bblocks_with_peaks
    >>> h_out = rebin_bblocks_with_peaks(h_fine, max_bin_width=10)
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
    """Rebin a fine histogram onto uniform continuum bins with collapsed peaks.

    Runs an internal :func:`rebin_bblocks` pass to find the peaks,
    then builds the output: uniform-``bin_width`` continuum bins
    everywhere except inside detected peak regions, which each
    become a single bin. Output edges are snapped to
    ``h.axes[0].edges``, so ``bin_width`` need not align exactly
    with the input grid.

    Parameters
    ----------
    h
        Input finely-binned 1D histogram.
    bin_width
        Continuum bin width (in axis units). To avoid tiny
        fragment bins right next to peaks, the continuum bin
        bordering each peak absorbs the leftover and can be up
        to ``2 * bin_width`` wide.
    n_sigma, width_factor, max_bin_width, p0
        See :func:`rebin_bblocks_with_peaks`.

    Returns
    -------
    out
        A ``Variable``-axis ``Weight``-storage histogram whose
        edges are a subset of the input edges.

    Examples
    --------
    >>> from pygama.math.rebin import rebin_uniform_with_peaks
    >>> import hist
    >>> h_fine = hist.new.Reg(1000, 0, 1000).Double().fill(...)
    >>> h_out = rebin_uniform_with_peaks(h_fine, bin_width=2, max_bin_width=10)
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
