"""Tests for the DPLMS helpers in ``pygama.pargen.dplms_ge_dict``."""

from __future__ import annotations

import numpy as np

from pygama.pargen.dplms_ge_dict import is_not_pile_up, signal_selection


def _reference_is_not_pile_up(peak_pos, peak_pos_neg, thr, lim, size):
    """Verbatim copy of the pre-vectorisation per-event loop."""
    bin_edges = np.linspace(size / 2 - lim, size / 2 + lim, 2 * lim)
    hist, bin_edges = np.histogram(peak_pos, bins=bin_edges)

    thr = thr * hist.max() / 100
    low_thr_idxs = np.where(hist[: hist.argmax()] < thr)[0]
    upp_thr_idxs = np.where(hist[hist.argmax() :] < thr)[0]

    idx_low = low_thr_idxs[-1] if low_thr_idxs.size > 0 else 0
    idx_upp = (
        upp_thr_idxs[0] + hist.argmax() if upp_thr_idxs.size > 0 else len(hist) - 1
    )

    llow, lupp = bin_edges[idx_low], bin_edges[idx_upp]

    idxs = []
    for n, nn in zip(peak_pos, peak_pos_neg, strict=False):
        condition1 = np.count_nonzero(n > 0) == 1
        condition2 = (
            np.count_nonzero((n > 0) & ((n < llow) | (n > lupp) & (n < size))) == 0
        )
        condition3 = np.count_nonzero(nn > 0) == 0
        idxs.append(condition1 and condition2 and condition3)
    return idxs, llow, lupp


def test_is_not_pile_up_matches_loop_reference():
    rng = np.random.default_rng(41)
    n_events, n_peaks, size, lim, thr = 500, 5, 762, 30, 0.1

    # zero-padded peak-position arrays: one main peak near the centre, with
    # random extra positive peaks and occasional negative-polarity peaks
    peak_pos = np.zeros((n_events, n_peaks))
    peak_pos[:, 0] = rng.normal(size / 2, 3, n_events)
    extra = rng.integers(0, size + 40, (n_events, n_peaks - 1))
    has_extra = rng.random((n_events, n_peaks - 1)) < 0.3
    peak_pos[:, 1:] = np.where(has_extra, extra, 0)

    peak_pos_neg = np.zeros((n_events, n_peaks))
    neg = rng.integers(0, size, (n_events, n_peaks))
    peak_pos_neg[:] = np.where(rng.random((n_events, n_peaks)) < 0.05, neg, 0)

    ref_idxs, ref_llow, ref_lupp = _reference_is_not_pile_up(
        peak_pos, peak_pos_neg, thr, lim, size
    )
    idxs, llow, lupp = is_not_pile_up(peak_pos, peak_pos_neg, thr, lim, size)

    assert llow == ref_llow
    assert lupp == ref_lupp
    assert np.array_equal(idxs, np.array(ref_idxs))
    # both selected and rejected events must be present for a meaningful test
    assert 0 < np.count_nonzero(idxs) < n_events


def test_signal_selection_is_deterministic():
    """signal_selection is pure in its inputs — the caching in dplms_ge_dict
    relies on this."""
    rng = np.random.default_rng(7)
    n_events, size = 200, 762

    class FakeCol:
        def __init__(self, nda):
            self.nda = nda

    peak_pos = np.zeros((n_events, 3))
    peak_pos[:, 0] = rng.normal(size / 2, 3, n_events)
    dsp_cal = {
        "peak_pos": FakeCol(peak_pos),
        "peak_pos_neg": FakeCol(np.zeros((n_events, 3))),
        "centroid": FakeCol(rng.normal(size / 2, 5, n_events)),
        "tp_90": FakeCol(rng.uniform(500, 900, n_events)),
        "tp_10": FakeCol(rng.uniform(100, 400, n_events)),
    }
    dplms_dict = {
        "rt_low": 96,
        "peak_lim": 30,
        "wsize": size,
        "bsize": 1024,
        "centroid_lim": 20,
        "dp_def": {"rt": 99, "pt": 0.1},
    }

    first = signal_selection(dsp_cal, dplms_dict, {})
    second = signal_selection(dsp_cal, dplms_dict, {"nm": 1, "za": 0})

    assert np.array_equal(first["idxs"], second["idxs"])
    for key in ("ct_ll", "ct_hh", "pp_ll", "pp_hh", "rt_ll", "rt_hh"):
        assert first[key] == second[key]
